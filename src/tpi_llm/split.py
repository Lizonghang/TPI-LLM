import os
import re
import json
import shutil
import torch
import numpy as np
import safetensors.torch
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from .modeling_utils import load_model_config
from .utils import (
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    QKVO_KEY_TEMPLATE,
    MLP_KEY_TEMPLATE,
    ROTARY_EMB_KEY_TEMPLATE,
    LAYERNORM_KEY_TEMPLATE,
    ATTN_SAVE_PATH,
    MLP_SAVE_PATH,
    INPUT_SAVE_PATH,
    OUTPUT_SAVE_PATH,
    INPUT_EMB_KEY,
    OUTPUT_LAYERNORM_KEY,
    OUTPUT_HEAD_KEY,
)


def validate_heads_per_node(heads_per_node, num_heads):
    """
    Validate heads_per_node for correctness.

    Parameters:
    - heads_per_node: List of integers representing the number of attention heads per node.
    - num_attention_heads: Total number of attention heads.

    Raises:
    - ValueError: If any value in heads_per_node is not positive or if the sum does not match num_heads.
    """
    if not all(isinstance(num_heads, int) for num_heads in heads_per_node):
        raise ValueError(f"All values in heads_per_node must be integers, but got {heads_per_node}.")

    if any(num_heads <= 0 for num_heads in heads_per_node):
        raise ValueError(f"All values in heads_per_node must be positive, but got {heads_per_node}.")

    if sum(heads_per_node) != num_heads:
        raise ValueError(
            f"Sum of heads_per_node {sum(heads_per_node)} does not match num_attention_heads={num_heads}.")


def get_heads_per_node(world_size, ratio, num_heads=None, num_kv_heads=None):
    """
    Calculate the distribution of attention heads across nodes.

    Args:
        world_size (int): The number of nodes.
        ratio (list of float): A list of ratios specifying how attention heads should be
            distributed across the nodes.
        num_heads (int, optional): The total number of attention heads.
        num_kv_heads (int, optional): The total number of key value attention heads.

    Returns:
        heads_per_node (list of int): A list where each element represents the number of
            attention heads assigned to each node.
        kv_heads_per_node (list of int): A list where each element represents the number of
            key value attention heads assigned to each node.
    """
    if num_heads is None and num_kv_heads is None:
        raise ValueError("Either num_heads or num_kv_heads must be specified, but not both.")
    validate_ratio(ratio, world_size)

    def _allocate_heads(num_heads_):
        heads_per_node_ = [int(num_heads_ * r) for r in ratio]
        difference_ = num_heads_ - sum(heads_per_node_)
        for i in range(abs(difference_)):
            heads_per_node_[-i-1] += 1
        return heads_per_node_

    kv_heads_per_node = _allocate_heads(num_kv_heads) if num_kv_heads is not None else []
    num_key_value_groups, remainder = divmod(num_heads, num_kv_heads)
    if remainder != 0:
        raise ValueError("The value of num_heads is not divisible by num_kv_heads.")
    heads_per_node = [h * num_key_value_groups for h in kv_heads_per_node]

    validate_heads_per_node(heads_per_node, num_heads)
    validate_heads_per_node(kv_heads_per_node, num_kv_heads)
    return heads_per_node, kv_heads_per_node


def validate_ratio(ratio, world_size):
    """
    Validate the given ratio.

    Parameters:
    - ratio: List of probability values representing the ratio of attention heads for each node.
    - world_size: Number of nodes.

    Raises:
    - ValueError: If the dimension of ratio does not match world_size or if ratio values are not
      positive or do not sum to 1.
    """
    if len(ratio) != world_size:
        raise ValueError(f"The dimension of ratio {len(ratio)} does not match world_size={world_size}.")

    if any(r <= 0 for r in ratio) or not abs(sum(ratio) - 1.0) < 1e-6:
        raise ValueError(f"Ratio values must be positive and sum to 1, but got {ratio} with sum {sum(ratio)}.")


def get_layers_in_sharded_weights(weight_map):
    """
    Groups the layer numbers from sharded weight files.

    Args:
        weight_map (dict): A dictionary where the keys are layer name and the values are
            the sharded weight filenames.

    Returns:
        dict: A dictionary where each key is the sharded weight filename, and each value
            is the set of layer numbers in the sharded weight file.
    """
    sharded_layers = defaultdict(set)
    for layer_name, bin_fn in weight_map.items():
        match = re.search(r'\d+', layer_name)
        if match:
            layer_num = int(match.group())
            sharded_layers[bin_fn].add(layer_num)
    return sharded_layers


def merge_and_load_weights(sharded_filenames, is_safetensors):
    """
    Merges and loads weights from multiple sharded files into a single state_dict.

    Parameters:
    - sharded_filenames (list of str): List of sharded weight filenames to be merged.
    - is_safetensors (bool): Whether the sharded files are in the SafeTensors format.

    Returns:
    - dict: A dictionary containing the merged weights from all sharded files.
    """
    merged_weights = {}
    for filename in sharded_filenames:
        if is_safetensors:
            sharded_weights = safetensors.torch.load_file(filename)
        else:
            sharded_weights = torch.load(filename, map_location="cpu")
        merged_weights.update(sharded_weights)
        del sharded_weights
    return merged_weights


def split_attention_heads(weights, layer_num, heads_per_node, kv_heads_per_node, head_dim, model_path, save_dir="split"):
    """
    Splits the attention head weights (Q, K, V, O) for a given layer and saves them into separate files
    based on the specified heads_per_node distribution.

    Parameters:
    - weights: The full model weights dictionary containing all layers.
    - layer_num: The specific layer number to process.
    - heads_per_node: A list specifying the number of attention heads allocated to each node.
    - kv_heads_per_node: A list specifying the number of key value attention heads allocated to each node.
    - head_dim: The dimension size of each attention head.
    - model_path: The path to the directory where the model files are stored.
    - save_dir: The subdirectory under model_path where the split weights will be saved. Default is "split".
    """
    # get weight keys for the current layer
    input_layernorm_key = LAYERNORM_KEY_TEMPLATE.format(l=layer_num, type="input")
    rotary_emb_key = ROTARY_EMB_KEY_TEMPLATE.format(l=layer_num)
    q_key = QKVO_KEY_TEMPLATE.format(l=layer_num, type="q")
    k_key = QKVO_KEY_TEMPLATE.format(l=layer_num, type="k")
    v_key = QKVO_KEY_TEMPLATE.format(l=layer_num, type="v")
    o_key = QKVO_KEY_TEMPLATE.format(l=layer_num, type="o")

    # split Q, K, V, O according to heads_per_node
    split_dims = (np.array(heads_per_node) * head_dim).tolist()
    split_dims_kv = (np.array(kv_heads_per_node) * head_dim).tolist()
    q_slices = weights[q_key].split(split_dims, dim=0)
    k_slices = weights[k_key].split(split_dims_kv, dim=0)
    v_slices = weights[v_key].split(split_dims_kv, dim=0)
    o_slices = weights[o_key].split(split_dims, dim=1)

    # save q_slices, k_slices, v_slices, o_slices to <save_dir>. For example, for node 1, layer 1,
    # and tensor q, k, v, o, the saved file should be <save_dir>/node_1/l1.qkvo.bin.
    for node_rank, (q_, k_, v_, o_) in enumerate(zip(q_slices, k_slices, v_slices, o_slices)):
        node_dir = os.path.join(model_path, save_dir, f"node_{node_rank}")
        os.makedirs(node_dir, exist_ok=True)

        # save q, k, v, o, rotary embedding, and input layernorm
        attn_path = os.path.join(node_dir, ATTN_SAVE_PATH.format(l=layer_num))
        state_dict = {
            input_layernorm_key: weights[input_layernorm_key],
            q_key: q_.clone(), k_key: k_.clone(), v_key: v_.clone(), o_key: o_.clone()
        }
        if rotary_emb_key in weights.keys():
            state_dict[rotary_emb_key] = weights[rotary_emb_key]
        torch.save(state_dict, attn_path)


def split_mlp(weights, layer_num, heads_per_node, model_path, save_dir="split"):
    """
    Splits the MLP weights (gate_proj, down_proj, up_proj) for a given Transformer layer
    and saves them into separate files based on the specified heads_per_node distribution.

    Parameters:
    - weights: The full model weights dictionary containing all layers.
    - layer_num: The specific layer number to process.
    - heads_per_node: A list specifying the number of attention heads allocated to each node.
    - model_path: The path to the directory where the model files are stored.
    - save_dir: The subdirectory under model_path where the split weights will be saved. Default is "split".
    """
    # get weight keys for the current layer
    post_attn_layernorm_key = LAYERNORM_KEY_TEMPLATE.format(l=layer_num, type="post_attention")
    gate_key = MLP_KEY_TEMPLATE.format(l=layer_num, type="gate")
    up_key = MLP_KEY_TEMPLATE.format(l=layer_num, type="up")
    down_key = MLP_KEY_TEMPLATE.format(l=layer_num, type="down")

    # calculate the dimensions to split the weights according to heads_per_node
    split_dims = [weights[gate_key].size(0) // len(heads_per_node)] * len(heads_per_node)
    for i in range(weights[gate_key].size(0) - sum(split_dims)):
        split_dims[-i-1] += 1
    assert sum(split_dims) == weights[gate_key].size(0)

    # split gate_proj, up_proj, down_proj based on split_dims
    gate_slices = weights[gate_key].split(split_dims, dim=0)
    up_slices = weights[up_key].split(split_dims, dim=0)
    down_slices = weights[down_key].split(split_dims, dim=1)

    # save gate_slices, up_slices, down_slices to <save_dir>. For example, for node 1, layer 1,
    # and tensor gate_proj, up_proj, down_proj, the saved file should be <save_dir>/node_1/l1.mlp.bin.
    for node_rank, (gate_, up_, down_) in enumerate(zip(gate_slices, up_slices, down_slices)):
        node_dir = os.path.join(model_path, save_dir, f"node_{node_rank}")
        os.makedirs(node_dir, exist_ok=True)
        mlp_path = os.path.join(node_dir, MLP_SAVE_PATH.format(l=layer_num))
        torch.save({
            post_attn_layernorm_key: weights[post_attn_layernorm_key],
            gate_key: gate_.clone(), up_key: up_.clone(), down_key: down_.clone()
        }, mlp_path)


def save_input_and_output_weights(weights, model_path, save_dir="split"):
    """
    This function is useful for saving the input embedding, output layernorm, and output head to
    the master node.

    Parameters:
    - weights: The full model weights dictionary containing all layers.
    - model_path: The path to the directory where the model files are stored.
    - save_dir: The subdirectory under model_path where the non-split weights will be saved. Default is "split".
    """
    # save input embedding and output layernorm, head on the master node.
    node_dir = os.path.join(model_path, save_dir, "node_0")
    os.makedirs(node_dir, exist_ok=True)

    # save input embedding
    if INPUT_EMB_KEY in weights.keys():
        save_path = os.path.join(node_dir, INPUT_SAVE_PATH)
        torch.save({INPUT_EMB_KEY: weights[INPUT_EMB_KEY]}, save_path)

    # save output layernorm and head
    if {OUTPUT_LAYERNORM_KEY, OUTPUT_HEAD_KEY}.issubset(set(weights.keys())):
        save_path = os.path.join(node_dir, OUTPUT_SAVE_PATH)
        torch.save({
            OUTPUT_LAYERNORM_KEY: weights[OUTPUT_LAYERNORM_KEY],
            OUTPUT_HEAD_KEY: weights[OUTPUT_HEAD_KEY],
        }, save_path)


def split_pretrained_model(model_path, world_size, ratio, save_dir="split"):
    """
    Splits the pretrained model parameters based on the specified ratio.

    Parameters:
    - model_path (str): Path to the directory containing the model files.
    - world_size (int): Number of nodes to distribute the attention heads across.
    - ratio (list of float): A list specifying the ratio of attention heads to distribute across nodes.
    - save_dir: The subdirectory under model_path where the split weights will be saved. Default is "split".
    """
    # Check if save_dir already exists, and delete it if it does
    shutil.rmtree(os.path.join(model_path, save_dir), ignore_errors=True)

    # load model configuration.
    config = load_model_config(model_path)
    num_hidden_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    hidden_size = config["hidden_size"]
    head_dim = hidden_size // num_heads

    # allocate attention heads according to ratio.
    heads_per_node, kv_heads_per_node = get_heads_per_node(
        world_size, ratio, num_heads=num_heads, num_kv_heads=num_kv_heads)

    bin_file = os.path.join(model_path, WEIGHTS_NAME)
    bin_index_file = os.path.join(model_path, WEIGHTS_INDEX_NAME)
    safetensors_file = os.path.join(model_path, SAFE_WEIGHTS_NAME)
    safetensors_index_file = os.path.join(model_path, SAFE_WEIGHTS_INDEX_NAME)
    is_safetensors = os.path.exists(safetensors_file) or os.path.exists(safetensors_index_file)

    if os.path.exists(bin_file) or os.path.exists(safetensors_file):
        # split pretrained model parameters with only one weight file
        if is_safetensors:
            full_weights = safetensors.torch.load_file(safetensors_file)
        else:
            full_weights = torch.load(bin_file, map_location="cpu")
    elif os.path.exists(bin_index_file) or os.path.exists(safetensors_index_file):
        # merge sharded files first and then split
        index_file = safetensors_index_file if is_safetensors else bin_index_file
        with open(index_file, "r") as f:
            index = json.loads(f.read())
        sharded_filenames = sorted(set(index["weight_map"].values()))
        sharded_filenames = [os.path.join(model_path, f) for f in sharded_filenames]
        full_weights = merge_and_load_weights(sharded_filenames, is_safetensors)
    else:
        raise NotImplementedError("Current weight files are not supported.")

    def _process_layer(layer_num):
        split_attention_heads(
            full_weights, layer_num, heads_per_node, kv_heads_per_node, head_dim, model_path, save_dir)
        split_mlp(full_weights, layer_num, heads_per_node, model_path, save_dir)

    # save input and output layer on the master node
    save_input_and_output_weights(full_weights, model_path, save_dir)
    with ThreadPoolExecutor(8) as executor:
        futures = [
            executor.submit(_process_layer, layer_num)
            for layer_num in range(num_hidden_layers)
        ]
        with tqdm(total=num_hidden_layers, desc="Slicing layer", leave=False) as pbar:
            for _ in as_completed(futures):
                pbar.update(1)  # update progress when a thread completes
