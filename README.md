# TPI-LLM: High-Performance Tensor Parallelism Inference System for Edge LLM Services.
TPI-LLM (Tensor Parallelism Inference for Large Language Models) is a LLM service system designed to bring LLM 
functions to resource-constrained edge networks. While cloud LLM services have achieved great success, privacy 
concerns arise and users do not want their conversations uploaded to the cloud as these conversations could 
involve sensitive personal information.

Our TPI-LLM system addresses the privacy issue by enabling LLM inference on edge devices with limited resources. 
The system leverages multiple edge devices to perform inference through tensor parallelism, combined with 
sophisticated memory window scheduling techniques to minimize memory usage. Currently, TPI-LLM can run the 
full-precision Llama-2-3B model on a single Mac with only 8GB of RAM, while maintaining a stable memory footprint 
below 0.7 GB. In the future, we will support larger models, such as Llama-3.1-70B and Llama-3.1-405B, across multiple edge 
devices, and introduce acceleration techniques to ensure efficient inference.

# Updates
* 2024/08/20: Add support for multi-host tensor parallelism.
* 2024/08/22: Add support for Llama 2, Llama 3 and Llama 3.1.
* 2024/08/26: Implement a file server to synchronize sliced model files to other nodes.
* 2024/08/27: Add the MXNET KVSTORE backend.

# Installation
## Use the Source Code
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

4. Install MXNET or NETSTORM:
```commandline
# To install NETSTORM, please see: https://github.com/fengwenjiao/netstorm
# To install MXNET:
> (tpi-llm) pip install mxnet==1.9.1
```

> **WARNING:** On macOS, the pre-built MXNet binaries do not have the KVSTORE module enabled by default.
> To use `dist_sync`, you should compile MXNET from source with `USE_DIST_KVSTORE=1` enabled.
> To build MXNET from source, following [this guidance](https://mxnet.apache.org/get_started/build_from_source).


## Using Pre-built Docker Image
We have provided Docker images for TPI-LLM, available on [Docker Hub](https://hub.docker.com/repository/docker/lizonghango00o1/tpi-llm/general). 
This is the easiest way to get started, as it includes all dependencies pre-installed.

```commandline
> docker run -dit --name master lizonghango00o1/tpi-llm:1.0.1.mx
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

## Run on Multiple Hosts
Assume we have 3 hosts, where we will run a scheduler and a server on host 1, and run 2 worker nodes on the 
other 2 hosts, respectively. Suppose their IP addresses are:

```text
IP of host 1: X.X.X.1 (scheduler and server)
IP of host 2: X.X.X.2 (master node)
IP of host 3: X.X.X.3
```

> It is ok to run the scheduler, (parameter) server, and master node on the same host.

The scheduler and server are used by KVSTORE to perform allreduce across the master node and the other worker 
nodes. The master node is regarded as the task publisher, who initiates the input prompt and display output
test stream to users, at the meantime it will slice the pretrained model weight files and serve as a file
server to send the sliced model weights to other worker nodes.

**Step 1**: To launch the scheduler and server, run the following commands on host 1:
```commandline
# Run the scheduler on host 1 (IP: X.X.X.1):
> DMLC_ROLE=scheduler DMLC_PS_ROOT_URI=X.X.X.1 DMLC_PS_ROOT_PORT=29500 DMLC_NUM_WORKER=2 DMLC_NUM_SERVER=1 python -c "import mxnet"

# Run the server on host 1 (IP: X.X.X.1):
> DMLC_ROLE=server DMLC_PS_ROOT_URI=X.X.X.1 DMLC_PS_ROOT_PORT=29500 DMLC_NUM_WORKER=2 DMLC_NUM_SERVER=1 python -c "import mxnet"
```

**Step 2:** To launch the master node, run the following command on host 2:
```commandline
# Run the master node on host 2 (IP: X.X.X.2, RANK = 0)
> DMLC_ROLE=worker DMLC_PS_ROOT_URI=X.X.X.1 DMLC_PS_ROOT_PORT=29500 DMLC_NUM_WORKER=2 DMLC_NUM_SERVER=1 \
    RANK=0 MASTER_ADDR=X.X.X.2 python examples/run_multihost.py --model_type llama --model_path <PATH-TO-MODEL-FILES> 
```

> **NOTE:** Please make sure that the scheduler, server, and master node can be accessed by all other nodes.

> **NOTE:** The master node also participate in tensor-parallelism computing.

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

**Step 3**: To launch other worker nodes, use the following command on other hosts (e.g., host 3):
```commandline
# Run the worker node on host 3 (IP: X.X.X.3, RANK > 0)
> DMLC_ROLE=worker DMLC_PS_ROOT_URI=X.X.X.1 DMLC_PS_ROOT_PORT=29500 DMLC_NUM_WORKER=2 DMLC_NUM_SERVER=1 \
    RANK=1 MASTER_ADDR=X.X.X.2 python examples/run_multihost.py --model_type llama --model_path <PATH-TO-MODEL-FILES>
```

The worker nodes will automatically download their model weight slices from the master node. If you have downloaded
the slices before, you can use the `--force_download` option to force a re-download.

## Run on Klonet
Coming soon.

## Optional Arguments
TPI-LLM provides several optional parameters that you can customize to control various aspects of the inference process. 
Below is a list of these options:

| Argument           | Default   | Type    | Description                                                                                          |
|--------------------|-----------|---------|------------------------------------------------------------------------------------------------------|
| `--prompt`         | `""`      | `str`   | The input prompt.                                                                                    |
| `--length`         | `20`      | `int`   | Maximum length of the generated sequence.                                                            |
| `--prefix`         | `""`      | `str`   | Text added prior to input for context.                                                               |
| `--use_gpu`        | `False`   | `bool`  | Whether to use GPU for inference. If false, use CPU by default.                                      |
| `--split_bin`      | `False`   | `bool`  | Split the pretrained model file. (available only on the master node)                                 |
| `--save_dir`       | `"split"` | `str`   | The directory to save split model files.                                                             |
| `--seed`           | `42`      | `int`   | Random seed for reproducibility.                                                                     |
| `--file_port`      | `29600`   | `int`   | Port number on the master node where the file server is listening on.                                |
| `--broadcast_port` | `29700`   | `int`   | Port number on the master node where auxiliary information is listening on.                          |
| `--force_download` | `False`   | `bool`  | Force worker nodes to re-download model weight slices. (available only on the non-master node)       |
| `--temperature`    | `1.0`     | `float` | Sampling temperature for text generation. (available only on the master node)                        |
| `--k`              | `0`       | `int`   | Number of highest probability tokens to keep for top-k sampling. (available only on the master node) |
| `--p`              | `0.9`     | `float` | Cumulative probability for nucleus sampling (top-p). (available only on the master node)             |
| `--memory_window`  | `2`       | `int`   | Size of the memory window used during inference. Should be at least 2.                               |