import os
import logging
import torch
from transformers import AutoTokenizer, TextStreamer
from tpi_llm import TPILlamaForCausalLM
from tpi_llm.split import split_pretrained_model
from tpi_llm.modeling_utils import load_model_config
from tpi_llm.distributed import (
    run_sync_server,
    download_file,
    CommunicatorBase,
    CommunicatorMaster,
    CommunicatorClient,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "llama": (TPILlamaForCausalLM, AutoTokenizer),
}
MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def adjust_length_to_model(length, max_sequence_length):
    """
    Adjusts the sequence length.

    Args:
        length (int): Desired sequence length.
        max_sequence_length (int): Maximum sequence length allowed by the model.

    Returns:
        int: Adjusted sequence length.
    """
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main(my_rank, args, dist=None):
    # set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed_all(args.seed)

    # split and synchronize pretrained model weights
    args.ratio = [1./args.world_size] * args.world_size  # todo: the decision of ratio should be optimized.
    split_file_path = os.path.join(args.model_path, args.save_dir)
    if my_rank == 0:  # for the master node
        if not os.path.exists(args.model_path):
            raise Exception(f"Model path {args.model_path} does not exist, "
                            f"please download the pretrained model parameters first.")

        # initialize communicator for master node
        comm = dist or CommunicatorMaster(args.master_ip, args.master_port, args.world_size)

        # split the pretrained model files if not split or forced to split
        if not os.path.exists(split_file_path) or args.split_bin:
            split_pretrained_model(
                model_path=args.model_path,
                world_size=args.world_size,
                ratio=args.ratio,
                save_dir=args.save_dir
            )
            logger.info(f"All weights are split and saved to {split_file_path}.")

        assert len(os.listdir(split_file_path)) == args.world_size, \
            f"We have {len(os.listdir(split_file_path))} weight slices but {args.world_size} nodes."

        # wait for other nodes to download sliced files
        run_sync_server(args.master_ip, args.file_port, args.model_path, split_file_path)
        # ensure the file download is executed after the master node binds its file port
        comm.barrier()
    else:  # for the non-master node
        comm = dist or CommunicatorClient(args.master_ip, args.master_port, my_rank)
        comm.barrier()
        # download sliced weight files from the master node
        if not os.path.exists(split_file_path) or args.force_download:
            os.makedirs(os.path.join(split_file_path, f"node_{my_rank}"), exist_ok=True)
            download_file(args.master_ip, args.file_port, my_rank, args.model_path, split_file_path)

    # load model configurations and set generation length
    model_config = load_model_config(args.model_path)
    max_seq_length = model_config.get("max_position_embeddings", 0)
    args.length = adjust_length_to_model(args.length, max_sequence_length=max_seq_length)
    args.device = "cuda" if args.use_gpu else "cpu"
    args.rank = my_rank
    assert args.memory_window >= 2, \
        "Memory window should be larger than 10."
    assert model_config.get("num_key_value_heads", 1e9) >= args.world_size, \
        "The number of nodes cannot be more than the number of kv heads."
    logger.info(f"My rank is {my_rank}, totally {args.world_size} nodes.")

    try:
        # select model and tokenizer
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError(f"Unsupported model type: {args.model_type}")

    # load model
    model = model_class.from_pretrained(args.model_path, comm, rank=my_rank, args=args)

    # the master node initializes tokenizer and encodes user prompt
    tokenizer, streamer = None, None
    input_ids = ""
    if my_rank == 0:
        tokenizer = tokenizer_class.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # set pad token if not set

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # encode the prompt text provided by the user
        prompt_text = args.prompt if args.prompt else input("User prompt >>> ")
        input_ids = tokenizer.encode(
            args.prefix + prompt_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(args.device)

    # generate output with streaming output
    model.generate(
        input_ids=input_ids,
        max_length=args.length + len(input_ids[0]) if my_rank == 0 else args.length,
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        do_sample=True,
        streamer=streamer,
        communicator=comm,
    )

    # stop running threads for a graceful exit
    model.mem_manager.stop()
    if isinstance(comm, CommunicatorBase):
        comm.close()
