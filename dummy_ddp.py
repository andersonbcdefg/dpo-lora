import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == "__main__":
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(f"cuda:{rank}")

    # return world size
    ws = torch.distributed.get_world_size()
    print("World size:", ws)