import os
import argparse
import torch.distributed as dist
from run_llama import main


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
    # hyperparameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--memory_window", type=int, default=2,
                        help="Memory window size, should be at least 2.")
    args = parser.parse_args()

    # retrieve world size and my rank from environment variables
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.rank = int(os.environ['RANK'])
    args.master_ip = os.environ['MASTER_ADDR']
    args.master_port = int(os.environ['MASTER_PORT'])

    # launch the distributed task
    dist.init_process_group(
        backend="gloo",
        init_method='env://',
        rank=args.rank,
        world_size=args.world_size
    )
    main(args.rank, args.world_size, args)
