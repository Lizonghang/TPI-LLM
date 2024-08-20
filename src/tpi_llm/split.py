import os
import shutil
import numpy as np
import torch
from .modeling_utils import load_model_config
from .utils import (
    WEIGHTS_NAME,
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


def validate_heads_per_node(heads_per_node, num_attention_heads):
    """
    Validate heads_per_node for correctness.

    Parameters:
    - heads_per_node: List of integers representing the number of attention heads per node.
    - num_attention_heads: Total number of attention heads.

    Raises:
    - ValueError: If any value in heads_per_node is not positive or if the sum does not match num_attention_heads.
    """
    if not all(isinstance(num_heads, int) for num_heads in heads_per_node):
        raise ValueError(f"All values in heads_per_node must be integers, but got {heads_per_node}.")

    if any(num_heads <= 0 for num_heads in heads_per_node):
        raise ValueError(f"All values in heads_per_node must be positive, but got {heads_per_node}.")

    if sum(heads_per_node) != num_attention_heads:
        raise ValueError(
            f"Sum of heads_per_node {sum(heads_per_node)} does not match num_attention_heads={num_attention_heads}.")


def get_heads_per_node(strategy=None, num_attention_heads=None, world_size=None, ratio=None):
    """
    Calculate the distribution of attention heads across nodes.

    Args:
        strategy (str): Set to "uniform" to allocate attention heads evenly.
        num_attention_heads (int): The total number of attention heads.
        world_size (int): The number of nodes.
        ratio (list of float, optional): A list of ratios specifying how attention heads
            should be distributed across the nodes. This argument is only used if `strategy`
            is not set to "uniform".

    Returns:
        heads_per_node (list of int): A list where each element represents the number of
            attention heads assigned to each node.
    """
    if strategy == "uniform":
        base_heads = num_attention_heads // world_size
        remainder = num_attention_heads % world_size
        heads_per_node = [base_heads] * world_size
        for node in range(remainder):
            heads_per_node[node] += 1
    elif ratio is not None:
        validate_ratio(ratio, world_size)
        heads_per_node = [int(num_attention_heads * r) for r in ratio]
        allocated_heads = sum(heads_per_node)
        if allocated_heads != num_attention_heads:
            difference = num_attention_heads - allocated_heads
            for i in range(abs(difference)):
                if difference > 0:
                    heads_per_node[i % world_size] += 1
                else:
                    heads_per_node[i % world_size] -= 1
    else:
        raise ValueError(f"Invalid strategy {strategy} with ratio=None.")

    # validate the attention heads distribution.
    validate_heads_per_node(heads_per_node, num_attention_heads)
    return heads_per_node


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


def split_attention_heads(weights, layer_num, heads_per_node, head_dim, model_path, save_dir="split"):
    """
    Splits the attention head weights (Q, K, V, O) for a given layer and saves them into separate files
    based on the specified heads_per_node distribution.

    Parameters:
    - weights: The full model weights dictionary containing all layers.
    - layer_num: The specific layer number to process.
    - heads_per_node: A list specifying the number of attention heads allocated to each node.
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
    q_slices = weights[q_key].split(split_dims, dim=0)
    k_slices = weights[k_key].split(split_dims, dim=0)
    v_slices = weights[v_key].split(split_dims, dim=0)
    o_slices = weights[o_key].split(split_dims, dim=1)

    # save q_slices, k_slices, v_slices, o_slices to <save_dir>. For example, for node 1, layer 1,
    # and tensor q, k, v, o, the saved file should be <save_dir>/node_1/l1.qkvo.bin.
    for node_rank, (q_, k_, v_, o_) in enumerate(zip(q_slices, k_slices, v_slices, o_slices)):
        node_dir = os.path.join(model_path, save_dir, f"node_{node_rank}")
        os.makedirs(node_dir, exist_ok=True)

        # save q, k, v, o, rotary embedding, and input layernorm
        attn_path = os.path.join(node_dir, ATTN_SAVE_PATH.format(l=layer_num))
        torch.save({
            input_layernorm_key: weights[input_layernorm_key],
            rotary_emb_key: weights[rotary_emb_key],
            q_key: q_, k_key: k_, v_key: v_, o_key: o_,
        }, attn_path)


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
    split_dims = (np.array(heads_per_node) * weights[gate_key].size(0) // sum(heads_per_node)).tolist()
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
            gate_key: gate_,
            up_key: up_,
            down_key: down_,
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
    save_path = os.path.join(node_dir, INPUT_SAVE_PATH)
    torch.save({INPUT_EMB_KEY: weights[INPUT_EMB_KEY]}, save_path)

    # save output layernorm and head
    save_path = os.path.join(node_dir, OUTPUT_SAVE_PATH)
    torch.save({
        OUTPUT_LAYERNORM_KEY: weights[OUTPUT_LAYERNORM_KEY],
        OUTPUT_HEAD_KEY: weights[OUTPUT_HEAD_KEY],
    }, save_path)


def split_pretrained_model(model_path, world_size, heads_per_node=None, strategy="uniform", ratio=None, save_dir="split"):
    """
    Splits the pretrained model parameters based on the specified strategy.

    Parameters:
    - model_path (str): Path to the directory containing the model files.
    - world_size (int): Number of nodes to distribute the attention heads across.
    - heads_per_node (list of int, optional): A list specifying the number of attention heads per node.
    - strategy (str, optional): If set to "uniform", allocate attention heads to each node equally.
    - ratio (list of float, optional): A list specifying the ratio of attention heads to distribute across nodes.
    - save_dir: The subdirectory under model_path where the split weights will be saved. Default is "split".
    """
    # Check if save_dir already exists, and delete it if it does
    shutil.rmtree(os.path.join(model_path, save_dir), ignore_errors=True)

    # load model configuration.
    config = load_model_config(model_path)
    num_hidden_layers = config["num_hidden_layers"]
    num_attention_heads = config["num_attention_heads"]
    hidden_size = config["hidden_size"]
    head_dim = hidden_size // num_attention_heads

    # allocate attention heads according to heads_per_node, or strategy, or ratio.
    if heads_per_node is None:
        heads_per_node = get_heads_per_node(strategy, num_attention_heads, world_size, ratio)

    if WEIGHTS_NAME in os.listdir(model_path):
        # split pretrained model parameters with only one file named "pytorch_model.bin"
        weight_file = os.path.join(model_path, WEIGHTS_NAME)
        full_weights = torch.load(weight_file)
        # save input and output layer on the master node
        save_input_and_output_weights(full_weights, model_path, save_dir)
        # split and save attention and mlp layers on each node
        for layer_num in range(num_hidden_layers):
            split_attention_heads(full_weights, layer_num, heads_per_node, head_dim, model_path, save_dir)
            split_mlp(full_weights, layer_num, heads_per_node, model_path, save_dir)
    else:
        raise NotImplementedError("Current weight files are not supported.")
