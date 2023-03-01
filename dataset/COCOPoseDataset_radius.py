import warnings, cv2, torch, math, pycocotools, json

import os.path as osp
from pycocotools.coco import COCO
import numpy as np
from torch.utils.data import Dataset

import dataset.transforms as T
from .target_generator import HeatmapGenerator
from .utils import imnormalize, impad_to_multiple, impad, imrescale, imresize, imflip


def batch_collate(batch):

    imgs, gt_heatmaps, inst_coords, inst_heatmaps, idxs = [], [], [], [], []
    masks = []
    for item in batch:
        imgs.append(item[0].unsqueeze(dim=0))
        gt_heatmaps.append(item[1].unsqueeze(dim=0))
        inst_coords.append(item[2])
        inst_heatmaps.append(item[3])
        idxs.append(item[4])
        masks.append(item[5].unsqueeze(dim=0))

    imgs = torch.cat(imgs, dim=0)
    gt_heatmaps = torch.cat(gt_heatmaps, dim=0)
    masks = torch.cat(masks, dim=0).unsqueeze(dim=1)

    return imgs, gt_heatmaps, inst_coords, inst_heatmaps, idxs, masks

def batch_collate_test(batch):

    imgs, img_metas, idx = [], [], []
    
    for item in batch:
        imgs.append(item[0].unsqueeze(dim=0))
        img_metas.append(item[1])
        idx.append(item[2])

    imgs = torch.cat(imgs, dim=0)
    #idx = torch.cat(idx, dim=0)
    return imgs, img_metas, idx

class COCOPoseDataset(Dataset):

    def __init__(self,
                 ann_file,
                 img_prefix,
                 input_size=512,
                 img_scale_test=(2000, 512),
                 test_mode=False):

        # prefix of images path
        self.img_prefix = img_prefix

        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())         

        cats = self.coco.loadCats(self.coco.getCatIds())
        keypoints_name = [cat['keypoints'] for cat in self.coco.cats.values()][0]
        print(len(keypoints_name), ' keypoints: ', keypoints_name)

        cat_ids = self.coco.getCatIds()
        self.cat2label = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
        #self.img_ids = self.coco.getImgIds() # 5000 for val
        self.test_mode = test_mode
        self._filter_imgs(test=test_mode)
        
        if test_mode:
            #img_norm_cfg= {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}
            self.test_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            self.test_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            self.img_scale_test = img_scale_test

        self.dataset = 'coco'
        if self.dataset == 'coco': #COCO
            self.keypoint_num = 17
            self.flip_index = [0, 2,1, 4,3, 6,5, 8,7, 10,9, 12,11, 14,13, 16,15]
        else: #CrowdPose
            self.keypoint_num = 14
            self.flip_index = [1,0, 3,2, 5,4, 7,6, 9,8, 11,10, 12, 13]

        
        self.output_size = input_size//4
        self.transforms = T.Compose([
            T.RandomAffineTransform(
                input_size=input_size,
                output_size=self.output_size,
                max_rotation=30,
                min_scale=0.75, 
                max_scale=1.5,                
                scale_type='short',
                max_translate=40),
            T.RandomHorizontalFlip(self.flip_index, output_size=self.output_size, prob=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.heatmap_generator = HeatmapGenerator(output_res=self.output_size)
        self.cpoints_idx = [[0, 1, 2, 3, 4], [5,6, 11,12], [11, 12, 13, 14, 15, 16]]

        self.angle_bins = 90
        self.radius = 2.0
        print('radius', self.radius)


    def __len__(self):
        return len(self.img_ids)

    def write_json(self, preds, scores, filenames, res_file):
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))

        cat_id = self._class_to_coco_ind[self.classes[1]] #person
        results = []
        for idx, key_points in enumerate(preds):
            if key_points.shape[0] == 0:
                continue

            key_points = key_points.astype(dtype=np.float)
            score = scores[idx].astype(dtype=np.float)            
            file_name = filenames[idx]

            key_points = key_points.reshape((-1, self.keypoint_num*3))
            key_points = key_points.astype(dtype=np.float)
            for k in range(key_points.shape[0]):

                kpt = key_points[k].reshape((self.keypoint_num, 3))                

                left_top = np.amin(kpt, axis=0)
                right_bottom = np.amax(kpt, axis=0)

                w = right_bottom[0] - left_top[0]
                h = right_bottom[1] - left_top[1]

                results.append({
                    'image_id': int(file_name.split('.')[0]),
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': score[k],
                    'bbox': list([left_top[0], left_top[1], w, h])
                })

        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _filter_imgs(self, test=False):
        """Filter images too small or without ground truths."""
        
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        print('ids_with_ann', len(ids_with_ann))

        self.img_ids = [i for i in self.img_ids if i in ids_with_ann]
        if test:
            print('valid ids test', len(self.img_ids))
            return

        #filter imgs with less 10 keypoints
        valid_inds2 = []
        for img_id in self.img_ids:
            
            ann_id = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            ann_info = self.coco.loadAnns(ann_id)

            num_keypoints_sum = 0
            for i, ann in enumerate(ann_info):
                num_keypoints_sum += ann['num_keypoints']

            if num_keypoints_sum > 10:
                valid_inds2.append(img_id)
        print('valid ids train', len(valid_inds2))

        self.img_ids = valid_inds2

    def get_ann_info(self, img_id):
        #img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann_info = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        #print(img_info)
        mask = np.zeros((img_info['height'], img_info['width']))
        gt_bboxes, gt_labels, gt_keypoints, areas = [], [], [], []
        for i, ann in enumerate(ann_info):

            if ann.get('ignore', False): continue
            #if ann['iscrowd'] or ann['num_keypoints']==0:
            if ann['iscrowd']:
                rle = pycocotools.mask.frPyObjects(ann['segmentation'], img_info['height'], img_info['width'])
                temp = pycocotools.mask.decode(rle)
                #print(mask.shape, temp.shape, temp.sum(), 'crowd')
                mask += temp#[:,:,0]                
            elif ann['num_keypoints']==0:
                rle = pycocotools.mask.frPyObjects(ann['segmentation'], img_info['height'], img_info['width'])
                temp = pycocotools.mask.decode(rle)
                #print(mask.shape, temp.shape, temp.sum(), 'num_keypoints0')
                mask += temp[:,:,0]

            if ann['iscrowd'] or ann['num_keypoints']==0: continue

            x1, y1, w, h = ann['bbox']
            keypoints = ann['keypoints']
            #num_keypoints = ann['num_keypoints']
            bbox = [[x1, y1],     [x1+w-1, y1],
                    [x1, y1+h-1], [x1+w-1, y1+h-1]]            

            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
            gt_keypoints.append(keypoints)
            #areas.append(ann['area'])
            areas.append(w*h)        
        #mask = (mask < 0.5).astype(np.float32)
        mask = mask<0.5

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            areas = np.array(areas, dtype=np.float32)
            gt_keypoints = np.array(gt_keypoints, dtype=np.float32)
            gt_keypoints = gt_keypoints.reshape((gt_keypoints.shape[0], -1, 3))
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            areas = np.zeros((0), dtype=np.float32)
            gt_keypoints = np.zeros((0, 17, 3), dtype=np.float32)

        #mask = mask<0.5
        

        ann = dict(
            bboxes=gt_bboxes, 
            labels=gt_labels,
            areas=areas,
            keypoints=gt_keypoints,
            mask=mask)

        return ann

    def prepare_train_img(self, idx):

        img_id = self.img_ids[idx]
        img_name = self.coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(osp.join(self.img_prefix, img_name), cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        
        ann = self.get_ann_info(img_id)
        bboxs = ann['bboxes']
        labels = ann['labels']
        keypoints = ann['keypoints']
        areas = ann['areas']
        mask=ann['mask']
        
        #print(img.shape, mask.shape, keypoints.shape, areas.shape, bboxs.shape)
        if self.transforms:
            img, mask, keypoints, areas, bboxs = self.transforms(img, mask, keypoints, areas, bboxs)
        #print('mask', mask.shape)

        num_person = keypoints.shape[0]
        num_cpoints = len(self.cpoints_idx)
        cpoints = np.zeros((num_person, num_cpoints, 3))
        for i in range(num_person):

            if not self.dataset == 'crowdpose':
                if areas[i] < 32 ** 2: continue

            for j, point_idx in enumerate(self.cpoints_idx):

                keypoints_select = keypoints[i, point_idx, :2]
                keypoints_vis = (keypoints[i, point_idx, 2:3] > 0).astype(np.float32)
                keypoints_sum = np.sum(keypoints_select * keypoints_vis, axis=0)
                keypoints_vis_count = keypoints_vis.sum()

                if keypoints_vis_count <= 0: cpoints[i, j, 2] = 0; continue
                cpoints[i, j, :2] = keypoints_sum / keypoints_vis_count
                cpoints[i, j, 2] = 2

        keypoints = np.concatenate((keypoints, cpoints), axis=1)
        gt_heatmaps = self.heatmap_generator(keypoints, bboxs)        
        
        #inst_coords, inst_heatmaps = [], []
        inst_coords, quantize_len_angles = [], []
        ind_vis = []
        area_idx = np.argsort(areas.squeeze())
        for i in area_idx:

            inst_coord = []
            for j in range(num_cpoints):

                cpoint = cpoints[i, j, :]
                if cpoint[2] == 0: continue

                cx, cy = int(cpoint[0]), int(cpoint[1])
                if cx < 0 or cx >= self.output_size: continue
                if cy < 0 or cy >= self.output_size: continue

                start_x = max(int(cx - self.radius), 0)
                start_y = max(int(cy - self.radius), 0)
                end_x = min(int(cx + self.radius), self.output_size)
                end_y = min(int(cy + self.radius), self.output_size)

                for x in range(start_x, end_x):
                    for y in range(start_y, end_y):

                        if [y, x, j] in ind_vis: continue

                        inst_coord.append([y, x, j])
                        ind_vis.append([y, x, j])
                        quantize_len_angle = self.get_len_angle(x, y, keypoints[i, 0:self.keypoint_num, :])
                        quantize_len_angles.append(quantize_len_angle)

            if len(inst_coord) == 0: continue

            inst_coords.append(np.array(inst_coord))


        if len(inst_coords) == 0: return None

        inst_coords = np.concatenate(inst_coords, axis=0)
        inst_coords = torch.from_numpy(inst_coords)#, dtype=torch.long)
        quantize_len_angles = np.stack(quantize_len_angles, axis=0)
        quantize_len_angles = torch.from_numpy(quantize_len_angles)
        gt_heatmaps = torch.from_numpy(gt_heatmaps)
        mask = torch.from_numpy(mask)
        #print(idx, gt_heatmaps.size(), inst_coords.size(), quantize_len_angles.size(), mask.size())

        return img, gt_heatmaps, inst_coords, quantize_len_angles, idx, mask


    def get_len_angle(self, c_x, c_y, pos_keypoint):

        lens = []
        angles = []
        #print(pos_keypoint.shape)
        for i in range(pos_keypoint.shape[0]):

            x = pos_keypoint[i,0]
            y = pos_keypoint[i,1]

            if x < 0 or y < 0 or x >= self.output_size or y >= self.output_size: 
                pos_keypoint[i,2]=0  

            if pos_keypoint[i,2]==0:
                lens.append(0.1)
                angles.append(0.0)
            else:
                dx = pos_keypoint[i,0]-c_x
                dy = pos_keypoint[i,1]-c_y
                
                if dx==0 and dy == 0:
                    lens.append(1.0)
                    angles.append(0.0)
                    continue

                angle = math.atan2(dy, dx)#+math.pi
                if dy < 0 and angle < 0:
                    angle = angle + math.pi*2

                angle = int(angle/(2*math.pi)*self.angle_bins+0.5)%self.angle_bins
                length = math.sqrt(dx**2+dy**2)

                lens.append(length)
                angles.append(float(angle))
                
                # angle_quantized = int(angle/(2*math.pi)*self.angle_bins+0.5)%self.angle_bins
                # length = math.sqrt(dx**2+dy**2)

                # angle_diff = float(angle_quantized)/self.angle_bins*math.pi*2 - angle
                # length = length * math.cos(angle_diff)

                # lens.append(length)
                # angles.append(float(angle_quantized))

        pose_target = torch.cat([torch.tensor(lens), torch.tensor(angles)])
        return pose_target



    def prepare_test_img(self, idx):

        img_id = self.img_ids[idx]
        img_name = self.coco.loadImgs(img_id)[0]['file_name']
        img_ori = cv2.imread(osp.join(self.img_prefix, img_name), 
            cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

        #cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB, img_ori)  # inplace
        ori_shape = img_ori.shape

        img = imnormalize(img_ori, self.test_mean, self.test_std, to_rgb=True)
        img, scale_factor = imrescale(img, self.img_scale_test, return_scale=True)
        img_shape = img.shape

        img = impad_to_multiple(img, divisor=32)
        pad_shape = img.shape            

        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor)

        img = img.transpose(2, 0, 1)
        img = torch.tensor(img)
        #imgs.append(img)
        #img_metas.append(img_meta)

        return img, img_meta, idx
        

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = np.random.choice(self.__len__())
                continue
            return data
