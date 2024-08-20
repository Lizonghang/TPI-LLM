import os
import argparse
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaTokenizer, TextStreamer
from src.tpi_llm import TPILlamaForCausalLM
from src.tpi_llm.split import split_pretrained_model
from src.tpi_llm.modeling_utils import load_model_config


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "llama": (TPILlamaForCausalLM, LlamaTokenizer),
}
MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main(my_rank, world_size, args):
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed_all(args.seed)
    model_config = load_model_config(args.model_path)
    max_seq_length = model_config.get("max_position_embeddings", 0)
    args.length = adjust_length_to_model(args.length, max_sequence_length=max_seq_length)
    args.device = "cuda" if args.use_gpu else "cpu"
    args.rank = my_rank
    assert args.memory_window >= 2, "Memory window should be larger than 10."
    logger.info(f"My rank is {my_rank}, running on device {args.device}.")

    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError(f"Unsupported model type: {args.model_type}")

    # todo: automatically determine ratio
    args.ratio = None

    # split pretrained model files on node with rank 0.
    if my_rank == 0 and args.split_bin:
        if not os.path.exists(args.model_path):
            raise Exception(f"Model path {args.model_path} does not exist, "
                            f"please download the pretrained model parameters first.")

        split_file_path = os.path.join(args.model_path, args.save_dir)
        split_pretrained_model(
            args.model_path,
            world_size,
            strategy=args.split_strategy,
            ratio=args.ratio,
            save_dir=args.save_dir
        )
        logger.info(f"All weights are splitted and saved to {split_file_path}.")
    else:
        os.makedirs(args.model_path, exist_ok=True)
        # todo: each node download sliced weight files from the master node.

    # load tokenizer and encode user prompt
    tokenizer, streamer = None, None
    input_ids = ""
    if my_rank == 0:
        tokenizer = tokenizer_class.from_pretrained(args.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        prompt_text = args.prompt if args.prompt else input("User prompt >>> ")
        input_ids = tokenizer.encode(
            args.prefix + prompt_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(args.device)

    # load model and run tensor-parallelism inference
    model = model_class.from_pretrained(args.model_path, rank=my_rank, args=args)

    # run generate with streaming output
    model.generate(
        input_ids=input_ids,
        max_length=args.length + len(input_ids[0]) if my_rank == 0 else args.length,
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        do_sample=True,
        streamer=streamer,
    )

    # print recorded memory usage
    time_str, mem_str = model.mem_manager.memory_history
    logger.info(f"RANK {my_rank}:\nTimestamp: {time_str} \nMemory usage: {mem_str}")


def init_process(rank, size, fn, args, backend="gloo"):
    os.environ["MASTER_ADDR"] = args.master_ip
    os.environ["MASTER_PORT"] = str(args.master_port)
    dist.init_process_group(backend, rank=rank, world_size=size)
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
    parser.add_argument("--split_strategy", type=str, default="uniform")
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
