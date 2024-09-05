import os
import re
import torch
import asyncio
from typing import Tuple, Deque
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from ..utils import (
    BLOCK_TEMPLATE,
    ATTN_SAVE_PATH,
    MLP_SAVE_PATH,
    INPUT_SAVE_PATH,
    OUTPUT_SAVE_PATH,
)


class MemoryManager:
    def __init__(self, model, rank, args):
        self._model = model
        self._device = model.device
        self._rank = rank
        self._split_dir = os.path.join(args.model_path, args.save_dir, f"node_{rank}")
        self._all_layers = set(model.state_dict().keys())
        self._all_blocks = ["input"] + [
            BLOCK_TEMPLATE.format(l=block_idx, type=block_type)
            for block_idx in range(model.config.num_hidden_layers)
            for block_type in ["self_attn", "mlp"]
        ] + ["output"]
        self._loaded_blocks: Deque[str] = deque(maxlen=args.memory_window)
        self._layers_in_block = {block_key: [] for block_key in self._all_blocks}
        self._disabled = args.disable_memory_schedule
        self._batch_loaded = False

    def _get_bid_and_btype(self, block_name: str) -> Tuple[int, str]:
        """
        Extracts the block id and block type from the given block name.

        Args:
            block_name (str): The name of the block, following the BLOCK_TEMPLATE pattern.

        Returns:
            Tuple[int, str]: A tuple containing the block id (int) and block type (str).

        """
        pattern = BLOCK_TEMPLATE.format(type=r'(\w+)', l=r'(\d+)')
        match = re.match(pattern, block_name)
        if match:
            return int(match.group(2)), match.group(1)
        else:
            raise ValueError(f"Key '{block_name}' does not match pattern '{pattern}'")

    def _find_module(self, model, key):
        module = model
        *module_names, param_name = key.split('.')
        for name in module_names:
            module = getattr(module, name, None)
            if module is None:
                raise ValueError(f"Parameter {key} not found.")
        return module

    def _load_block_until_filled(self, block_name: str):
        """
        Loads multiple blocks starting from the given block until self._loaded_blocks is full.

        Args:
            block_name (str): The starting block name to load.
        """
        # ensure the block_name exists
        if block_name not in self._all_blocks:
            raise ValueError("Block name {} is not valid.".format(block_name))

        if self._disabled and self._batch_loaded:
            return

        # get the starting index of block_name
        start_idx = self._all_blocks.index(block_name)

        # load blocks sequentially until the deque is full
        for idx in range(start_idx, len(self._all_blocks)):
            block_name_ = self._all_blocks[idx]

            # skip if the block is already loaded
            if block_name_ in self._loaded_blocks:
                continue

            # append the loaded block name to the deque
            self._loaded_blocks.append(block_name_)

            # determine the path to the binary file
            if block_name_ == "input":
                bin_path = os.path.join(self._split_dir, INPUT_SAVE_PATH)
            elif block_name_ == "output":
                bin_path = os.path.join(self._split_dir, OUTPUT_SAVE_PATH)
            elif "self_attn" in block_name_:
                block_id, _ = self._get_bid_and_btype(block_name_)
                bin_path = os.path.join(self._split_dir, ATTN_SAVE_PATH.format(l=block_id))
            elif "mlp" in block_name_:
                block_id, _ = self._get_bid_and_btype(block_name_)
                bin_path = os.path.join(self._split_dir, MLP_SAVE_PATH.format(l=block_id))
            else:
                raise NotImplementedError(f"Block name {block_name_} is not supported.")

            # load pretrained weights into model tensors
            try:
                with open(bin_path, 'rb') as f:
                    pretrained_weights = torch.load(f, map_location=self._device)

                for key, weight in pretrained_weights.items():
                    if key in self._all_layers:
                        *module_names, param_name = key.split('.')
                        module = self._find_module(self._model, key)
                        module.register_parameter(
                            param_name,
                            torch.nn.Parameter(weight.float(), requires_grad=False)
                        )
                        if key not in self._layers_in_block[block_name_]:
                            self._layers_in_block[block_name_].append(key)

                del pretrained_weights
            except FileNotFoundError:
                if block_name_ != "output":
                    raise FileNotFoundError(f"Binary file {bin_path} not found.")

            # stop if the deque is full
            if not self._disabled and len(self._loaded_blocks) == self._loaded_blocks.maxlen:
                break

        if self._disabled:
            self._batch_loaded = True

    def _release_block(self, block_name: str):
        """
        Releases the memory of the tensors associated with the specified block.

        Args:
            block_name (str): The name of the block to release.
        """
        if block_name not in self._layers_in_block:
            raise KeyError(f"Block name '{block_name}' not found in _layers_in_block.")

        for layer_key in self._layers_in_block[block_name]:
            module = self._find_module(self._model, layer_key)
            *module_names, param_name = layer_key.split('.')
            param = module._parameters[param_name]
            if param.device.type == "cuda":
                with torch.no_grad():
                    param.data = None
                torch.cuda.empty_cache()
            else:
                del param

    def track(self, block_name: str, async_op: bool = False) -> Future:
        """
        Starts a background task to schedule the loading and releasing of blocks.

        Args:
            block_name (str): The name of the block currently processing.
            async_op (bool, optional): Whether the task is asynchronous or not. Defaults to False.

        Returns:
            Future: A concurrent.futures.Future object representing the asynchronous task.
                Use self.wait() to wait for the background task to complete.
        """

        def _track_func(block_name_: str):
            if block_name_ not in self._all_blocks:
                return

            # release all blocks before this block
            while not self._disabled and self._loaded_blocks:
                current_block = self._loaded_blocks[0]
                if (block_name_ == "input" or
                        self._all_blocks.index(current_block) < self._all_blocks.index(block_name_)):
                    self._release_block(self._loaded_blocks.popleft())
                else:
                    break

            # load the block and subsequent blocks until the deque is full
            self._load_block_until_filled(block_name)

        # Create a thread pool executor and run track function in the background using a thread pool
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        track_thread = loop.run_in_executor(executor, _track_func, block_name)
        if not async_op: self.wait(track_thread)
        return track_thread

    def wait(self, thread: Future) -> any:
        """
        Waits for the result of the given background task.

        Args:
            thread (Future): A concurrent.futures.Future object representing the background task.

        Returns:
            any: The result of the background task.
        """
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(thread)
        return result
