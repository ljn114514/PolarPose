#import test
import os
import socket
import torch.distributed as dist

def find_free_port() -> int:
	"""
	Used for distributed learning
	"""
	
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sock.bind(("", 0))
	port = sock.getsockname()[1]
	sock.close()
	return port

def setup(rank, world_size, port):
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = str(port)
	# initialize the process group
	dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
	dist.destroy_process_group()