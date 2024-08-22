import os
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from run_llama import main


def init_process(rank, size, fn, args, backend="gloo"):
    os.environ["MASTER_ADDR"] = args.master_ip
    os.environ["MASTER_PORT"] = str(args.master_port)
    dist.init_process_group(backend, init_method='env://', rank=rank, world_size=size)
    fn(rank, size, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # necessary arguments
    parser.add_argument("--model_type", default=None, type=str, required=True)
    parser.add_argument("--model_path", default=None, type=str, required=True)
    parser.add_argument("--world_size", default=None, type=int, required=True)
    # arguments with default values
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--use_gpu", action="store_true", help="Whether to use gpu, default to use cpu.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit mixed precision.")
    parser.add_argument("--split_bin", action="store_true", help="Whether to split the model file.")
    parser.add_argument("--save_dir", type=str, default="split", help="Directory to save split models.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--master_ip", type=str, default="127.0.0.1", help="Master IP address.")
    parser.add_argument("--master_port", type=int, default=29500, help="Master port.")
    # hyperparameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--memory_window", type=int, default=2,
                        help="Memory window size, should be at least 2.")
    args = parser.parse_args()

    processes = []
    mp.set_start_method("spawn")
    for rank in reversed(range(args.world_size)):
        p = mp.Process(target=init_process, args=(rank, args.world_size, main, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
