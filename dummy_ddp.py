import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"]) 
    world_size = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:12345',
        rank=local_rank,
        world_size=world_size
    )
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(f"cuda:{rank}")

    # return world size
    ws = torch.distributed.get_world_size()
    print("World size:", ws)