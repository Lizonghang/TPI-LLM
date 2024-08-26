import os
import logging
import argparse
import torch.distributed as dist
# from mxnet import kv
from run_llama import main

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # necessary arguments
    parser.add_argument("--model_type", default=None, type=str, required=True)
    parser.add_argument("--model_path", default=None, type=str, required=True)
    # arguments with default values
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--use_gpu", action="store_true", help="Whether to use gpu, default to use cpu.")
    parser.add_argument("--split_bin", action="store_true", help="Whether to split the model file.")
    parser.add_argument("--save_dir", type=str, default="split", help="Directory to save split models.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    # for model synchronization
    parser.add_argument("--file_port", type=int, default=29600, help="File server port.")
    parser.add_argument("--force_download", action="store_true", help="Force download sliced model files.")
    # hyperparameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--memory_window", type=int, default=2,
                        help="Memory window size, should be at least 2.")
    args = parser.parse_args()

    # launch kvstore backend
    # kvstore_dist = kv.create("dist_sync")
    # args.rank = kvstore_dist.rank
    # args.world_size = kvstore_dist.num_workers
    # logger.info(f"KVStore initialized with rank {args.rank} / {args.world_size}")

    # launch torch distributed backend
    args.master_ip = os.environ["MASTER_ADDR"]
    args.master_port = int(os.environ["MASTER_PORT"])
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    # os.environ["WORLD_SIZE"] = str(args.world_size)
    # os.environ["RANK"] = str(args.rank)
    logger.info("call dist.init_process_group")
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=args.rank,
        world_size=args.world_size
    )
    logger.info(f"PyTorch distributed initialized with rank {args.rank} / {args.world_size}")

    # run inference
    # main(kvstore_dist, args.rank, args.world_size, args)

    dist.barrier()
    dist.destroy_process_group()
