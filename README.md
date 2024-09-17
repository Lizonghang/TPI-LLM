# TPI-LLM: High-Performance Tensor Parallelism Inference System for Edge LLM Services.
TPI-LLM (Tensor Parallelism Inference for Large Language Models) is a LLM service system designed to bring LLM 
functions to resource-constrained edge networks. While cloud LLM services have achieved great success, privacy 
concerns arise and users do not want their conversations uploaded to the cloud as these conversations could 
involve sensitive personal information.

Our TPI-LLM system addresses the privacy issue by enabling LLM inference on edge devices with limited resources. 
The system leverages multiple edge devices to perform inference through tensor parallelism, combined with 
sophisticated memory window scheduling techniques to minimize memory usage. Currently, TPI-LLM can run the 
full-precision Llama-3.1-8B model on 2 laptops with 8GB of RAM. In the future, we will introduce more techniques 
to speed up LLM inference.

# Updates
* 2024/08/20: Add support for multi-host tensor parallelism.
* 2024/08/22: Add support for Llama 2, Llama 3 and Llama 3.1.
* 2024/08/26: Implement a file server to synchronize sliced model files to other nodes.
* 2024/09/07: Add support for deamon memory scheduling, use `--disable_memory_schedule` to disable this feature.

# Installation
## Use the Source Code
1. Clone the repository:
```commandline
> git clone https://github.com/Lizonghang/TPI-LLM
> cd TPI-LLM
> git checkout tpi-mx
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

## Using Pre-built Docker Image
We have provided Docker images for TPI-LLM, available on [Docker Hub](https://hub.docker.com/repository/docker/lizonghango00o1/tpi-llm/general). 
This is the easiest way to get started, as it includes all dependencies pre-installed.

```commandline
> docker run -dit --name master lizonghango00o1/tpi-llm:1.0.3.mx
```

If this is a master node, use `docker cp <HOST_MODEL_PATH> master:/root/TPI-LLM/` to copy the pretrained model files
to the container of the master node.

## Build from Dockerfile
If you prefer to build the Docker image yourself, you can modify and use the provided 
[Dockerfile](https://github.com/Lizonghang/TPI-LLM/blob/tpi-mx/Dockerfile) in our repo.

```commandline
> docker build -t tpi-llm:local .
> docker run -dit --name master tpi-llm:local
```

# How to Use?

## Download Pretrained Model Weights

To get started, you’ll need to download the pretrained model weights from **Hugging Face**:

- **Llama 2 series**, for example, [Meta/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- **Llama 3 series**, for example, [Meta/Llama-3-8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main)
- **Llama 3.1 series**, for example, [Meta/Llama-3.1-8b-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

After downloading, save the model files in a directory of your choice, which we’ll refer to as `<PATH-TO-MODEL-FILES>`.

## Run on Your Laptop
Run the example script for a trial:

```commandline
> python examples/run_multiprocess.py --world_size 4 --model_type llama --model_path <PATH-TO-MODEL-FILES>
```

This command will run 4 processes on a single machine, creating a pseudo-distributed environment that leverages tensor parallelism for Llama inference.

_First-Time Setup:_

If this is your first time running the task, the master node will automatically slice the pretrained model weight 
files. Suppose we have 4 worker nodes (including the master node), the sliced model weight files should be like 
the following:

```commandline
> ls <PATH-TO-MODEL-FILES>
|- config.json
|- model-00001-of-00004.safetensors
|- model-00002-of-00004.safetensors
|- model-00003-of-00004.safetensors
|- model-00004-of-00004.safetensors
|- model.safetensors.index.json
|- ...
|- split/
|--- node_0
|--- node_1
|--- node_2
|--- node_3
```

_Subsequent Runs:_

For subsequent runs, the sliced model weight files can be reused. Or you can include the `--split_bin` option 
to re-split it.

## Run on Multiple Hosts
Assume we have 2 hosts with IP addresses as follows:

```text
IP of host 1: 172.17.0.3 (master node)
IP of host 2: 172.17.0.4 (worker node)
```

The master node is regarded as the task publisher, who initiates the input prompt and display output
test stream to users, at the meantime it will slice the pretrained model weight files and serve as a file
server to send the sliced model weights to other worker nodes.

**Step 1:** To launch the master node, run the following command on host 1:
```commandline
# Run the master node on host 1 (IP: 172.17.0.3, RANK = 0)
> python examples/run_multihost.py --rank 0 --world_size 2 --master_ip 172.17.0.3 --master_port=29500 --model_type llama --model_path <PATH-TO-MODEL-FILES>
```

> **NOTE:** Please make sure the master node can be accessed by all other nodes. The master node also participate in tensor-parallel inference.

**Step 2**: To launch other worker nodes, use the following command on other hosts (e.g., host 2):
```commandline
# Run the worker node on host 2 (IP: 172.17.0.4, RANK > 0)
> python examples/run_multihost.py --rank 1 --world_size 2 --master_ip 172.17.0.3 --master_port=29500 --model_type llama --model_path <PATH-TO-MODEL-FILES>
```

The worker nodes will automatically download their model weight slices from the master node. If you have downloaded
the slices before, you can use the `--force_download` option to force a re-download.

## Run on Klonet
Coming soon.

## Other Arguments
TPI-LLM provides several optional parameters that you can customize to control various aspects of the inference process. 
Below is a list of these options:

| Argument                    | Default   | Type    | Description                                                                                          |
|-----------------------------|-----------|---------|------------------------------------------------------------------------------------------------------|
| `--prompt`                  | `""`      | `str`   | The input prompt.                                                                                    |
| `--length`                  | `20`      | `int`   | Maximum length of the generated sequence.                                                            |
| `--prefix`                  | `""`      | `str`   | Text added prior to input for context.                                                               |
| `--split_bin`               | `False`   | `bool`  | Split the pretrained model file. (available only on the master node)                                 |
| `--save_dir`                | `"split"` | `str`   | The directory to save split model files.                                                             |
| `--seed`                    | `42`      | `int`   | Random seed for reproducibility.                                                                     |
| `--file_port`               | `29600`   | `int`   | Port number on the master node where the file server is listening on.                                |
| `--force_download`          | `False`   | `bool`  | Force worker nodes to re-download model weight slices. (available only on the non-master node)       |
| `--temperature`             | `1.0`     | `float` | Sampling temperature for text generation. (available only on the master node)                        |
| `--k`                       | `0`       | `int`   | Number of highest probability tokens to keep for top-k sampling. (available only on the master node) |
| `--p`                       | `0.9`     | `float` | Cumulative probability for nucleus sampling (top-p). (available only on the master node)             |
| `--disable_memory_schedule` | `False`   | `bool`  | Set to True to disable memory window scheduling, this may lead to higher speed.                      |
| `--memory_window`           | `2`       | `int`   | Size of the memory window used during inference. Should be at least 2.                               |
| `--torch_dist`              | `False`   | `bool`  | Whether to use torch.distributed.                                                                    |