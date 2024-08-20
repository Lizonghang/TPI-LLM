# TPI-LLM: High-Performance Tensor Parallelism Inference System for Edge LLM Services.
TPI-LLM (Tensor Parallelism Inference for Large Language Models) is a LLM service system designed to bring LLM 
functions to resource-constrained edge networks. While cloud LLM services have achieved great success, privacy 
concerns arise and users do not want their conversations uploaded to the cloud as these conversations could 
involve sensitive personal information.

Our TPI-LLM system addresses the privacy issue by enabling LLM inference on edge devices with limited resources. 
The system leverages multiple edge devices to perform inference through tensor parallelism, combined with 
sophisticated memory window scheduling techniques to minimize memory usage. Currently, TPI-LLM can run the 
full-precision Llama-2-3B model on a single Mac with only 8GB of RAM, while maintaining a stable memory footprint 
below 0.7GB. In the future, we will support larger models, such as Llama-3-8B and Llama-3-70B, across multiple edge 
devices, and introduce acceleration techniques to ensure efficient inference.

# Updates
* 2024/08/20: Add support for multi-host tensor parallelism.

# Installation
1. Clone the repository:
```commandline
> git clone https://github.com/Lizonghang/TPI-LLM
> cd TPI-LLM
```

2. Add `PYTHONPATH` to `.bashrc`:
```commandline
> vim ~/.bashrc

# Set PYTHONPATH to the TPI-LLM/src folder
export PYTHONPATH=<PATH-TO-TPI-LLM>/src
```

3. Create a new conda environment and install dependencies:
```commandline
> conda create -n tpi-llm python=3.9
> conda activate tpi-llm
(tpi-llm) > pip install -r requirements.txt
```

# How to Use?

**1. Download Pretrained Model Weights**

To get started, you’ll need to download the pretrained model weights from Hugging Face. For example, if you’re 
using the Llama-2-3B model, you can download the weights manually from [Hugging Face](https://huggingface.co/openlm-research/open_llama_3b_v2). 
After downloading, save the model files in a directory of your choice, which we’ll refer to as `<PATH-TO-MODEL-FILES>`.

## Run on Your Laptop
Run the example script for a trial:
```commandline
> python examples/run_multiprocess.py \
    --model_type llama \
    --model_path <PATH-TO-MODEL-FILES> \
    --world_size 4 \
    --length 10 \
    --split_bin
```
This command will run 4 processes on a single machine, creating a pseudo-distributed environment that leverages 
tensor parallelism for Llama inference.

**First-Time Setup:**

If this is your first time running the script, make sure to include the <code>--split_bin</code> option. 
This will slice the pretrained model weights and save them into subdirectories corresponding to the 4 nodes:


```commandline
> ls <PATH-TO-MODEL-FILES>
|- pytorch_model.bin
|- config.json
|- ...
|- split/
|--- node0
|--- node1
|--- node2
|--- node3
```

**Subsequent Runs:**

For subsequent runs, you can omit the <code>--split_bin</code> option, as the model weights will already be sliced 
and saved in the respective node directories.

## Run on Multiple Hosts
Assume we have four hosts 0 ~ 3. Run the following command on each of them:

```commandline
# On node 0:
> RANK=0 WORLD_SIZE=4 python examples/run_multihost.py \
    --model_type llama \
    --model_path <PATH-TO-MODEL-FILES> \
    --length 10 \
    --master_ip <MASTER_ADDR> \
    --master_port <MASTER_PORT> \
    --split_bin

# On node 1:
> RANK=1 WORLD_SIZE=4 python examples/run_multihost.py \
    --model_type llama \
    --model_path <PATH-TO-MODEL-FILES> \
    --length 10 \
    --master_ip <MASTER_ADDR> \
    --master_port <MASTER_PORT> \
    --split_bin

# On node 2:
> RANK=2 WORLD_SIZE=4 python examples/run_multihost.py \
    --model_type llama \
    --model_path <PATH-TO-MODEL-FILES> \
    --length 10 \
    --master_ip <MASTER_ADDR> \
    --master_port <MASTER_PORT> \
    --split_bin
    
# On node 3:
> RANK=3 WORLD_SIZE=4 python examples/run_multihost.py \
    --model_type llama \
    --model_path <PATH-TO-MODEL-FILES> \
    --length 10 \
    --master_ip <MASTER_ADDR> \
    --master_port <MASTER_PORT> \
    --split_bin
```

You can set `<MASTER_ADDR>` and `<MASTER_PORT>` of your choice, but make sure that the master node can be accessed 
by all other nodes.

## Run on Klonet
Coming soon.

## Optional Arguments
TPI-LLM provides several optional parameters that you can customize to control various aspects of the inference process. 
Below is a list of these options:

| Argument           | Default       | Type    | Description                                                            |
|--------------------|---------------|---------|------------------------------------------------------------------------|
| `--prompt`         | `""`          | `str`   | The input prompt.                                                      |
| `--length`         | `20`          | `int`   | Maximum length of the generated sequence.                              |
| `--prefix`         | `""`          | `str`   | Text added prior to input for context.                                 |
| `--use_gpu`        | `False`       | `bool`  | Whether to use GPU for inference. If false, use CPU by default.        |
| `--split_bin`      | `False`       | `bool`  | Split the pretrained model file.                                       |
| `--save_dir`       | `"split"`     | `str`   | The directory to save split model files.                               |
| `--split_strategy` | `"uniform"`   | `str`   | Strategy for splitting the model across nodes.                         |
| `--seed`           | `42`          | `int`   | Random seed for reproducibility.                                       |
| `--master_ip`      | `"127.0.0.1"` | `str`   | IP address of the master node.                                         |
| `--master_port`    | `29500`       | `int`   | Port number of the master node.                                        |
| `--temperature`    | `1.0`         | `float` | Sampling temperature for text generation.                              |
| `--k`              | `0`           | `int`   | Number of highest probability tokens to keep for top-k sampling.       |
| `--p`              | `0.9`         | `float` | Cumulative probability for nucleus sampling (top-p).                   |
| `--memory_window`  | `2`           | `int`   | Size of the memory window used during inference. Should be at least 2. |