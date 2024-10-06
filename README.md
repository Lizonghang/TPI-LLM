# TPI-LLM: Serving 70b-scale LLMs Efficiently on Low-resource Edge Devices
TPI-LLM (Tensor Parallelism Inference for Large Language Models) is a LLM service system designed to bring LLM 
functions to low-resource edge devices. While cloud LLM services have achieved great success, privacy 
concerns arise and users do not want their conversations uploaded to the cloud as these conversations could 
involve sensitive personal information.

Our TPI-LLM system addresses the privacy issue by enabling LLM inference on edge devices with limited resources. 
The system leverages multiple edge devices to perform inference through tensor parallelism, combined with 
a sliding window memory scheduler to minimize memory usage. Currently, TPI-LLM can run Yi-34B in
full precision on 4 laptops with 5GB of memory on each laptop, and run Llama 2-70B on 8 devices 
with 3GB of memory on each device. Furthermore, TPI-LLM has demonstrated over 80% less TTFT and token latency
compared to [Accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling), 
over 90% compared to [Transformers](https://github.com/huggingface/transformers) and 
[Galaxy](https://github.com/ysyisyourbrother/Galaxy-LM), and 50\%-70\% compared to 
[llama.cpp](https://github.com/ggerganov/llama.cpp) on larger models (>13B).

| **Model (FP32)** | **Transformers** | **Accelerate** | **llama.cpp (INT4)** | **llama.cpp (INT8)** | **Transformers (with our MS)** | **TPI-LLM (Klonet, 8 devices, wire connected)** | **TPI-LLM (Home, 4 laptops, wireless connected)** |
|------------------|------------------|----------------|----------------------|----------------------|--------------------------------|-------------------------------------------------|---------------------------------------------------|
| **Llama 2-3B**   | 30 s/token       | 16 s/token     | **0.05** s/token     | 0.07 s/token         | 3 s/token                      | 2 s/token                                       | 2 s/token                                         |
| **Llama 2-7B**   | 56 s/token       | 26 s/token     | **0.08** s/token     | 8 s/token            | 8 s/token                      | 3 s/token                                       | 5 s/token                                         |
| **Llama 3.1-8B** | 65 s/token       | 31 s/token     | **1** s/token        | 11 s/token           | 12 s/token                     | 4 s/token                                       | 8 s/token                                         |
| **Llama 2-13B**  | OOM              | OOM            | 10 s/token           | 20 s/token           | 18 s/token                     | **3** s/token                                   | 9 s/token                                         |
| **Yi-34B**       | OOM              | OOM            | 29 s/token           | 51 s/token           | 55 s/token                     | **14** s/token                                  | 29 s/token                                        |

*Note: We set up two testbeds: the home testbed (4 laptops connected via local Wi-Fi) and the Klonet testbed (8 devices connected via a wire edge network).*

*Note: Computations were in **full precision** on **solely CPUs**, except for llama.cpp, which used Apple Metal Graphics and INT4/INT8 quantization for acceleration.*

*Note: Except for TPI-LLM, all other benchmarks were run on a Mac M1 laptop with 8 cores and 8GB memory.*

In the future, we plan to migrate to llama.cpp and add supports for Q4/Q8 quantizations and integrated GPUs.

# Installation
## Use the Source Code
1. Clone this repo and enter the project folder.

2. Add `PYTHONPATH` to `.bashrc`:
```commandline
> vim ~/.bashrc
export PYTHONPATH=<PATH-TO-TPI-LLM>/src
```

3. Create a new conda environment and install dependencies:
```commandline
> conda create -n tpi-llm python=3.9
> conda activate tpi-llm
(tpi-llm) > pip install -r requirements.txt
```

## Using Pre-built Docker Image
We provide Docker images for TPI-LLM, available on [Docker Hub](https://hub.docker.com/repository/docker/lizonghango00o1/tpi-llm/general). 
This is the easiest way to get started, but the container may slow down inference speed.

If the container is a master node, use `docker cp <HOST_MODEL_PATH> master:/root/TPI-LLM/` to copy the pretrained model files
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
- **01 AI Yi series**, for example, [chargoddard/Yi-34B-Llama](https://huggingface.co/chargoddard/Yi-34B-Llama)

Please make sure that the downloaded weight files conform to the HuggingFace format.

After downloading, save the model files in a directory of your choice, which we’ll refer to as `/root/TPI-LLM/pretrained_models/Llama-2-1.1b-ft`.

## Run on Your Laptop
Run the example script for a trial:

```commandline
> python examples/run_multiprocess.py --world_size 4 --model_type llama --model_path /root/TPI-LLM/pretrained_models/Llama-2-1.1b-ft --prompt "how are you?" --length 20 --memory_window 4
```

This command will run 4 processes on a single machine, creating a pseudo-distributed environment that leverages tensor parallelism for Llama inference.

_First-Time Setup:_

If this is your first time running the task, the master node will automatically slice the pretrained weight 
files. Suppose we have 4 worker nodes (including the master node), the sliced weight files should be like 
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
Assume we have 2 laptops with IP addresses as follows:

```text
IP of host 1: 192.168.2.1 (master node)
IP of host 2: 192.168.2.2 (worker node)
```

The master node is regarded as the task publisher, who initiates the prompt and display generated text to users, 
it also slices the pretrained weight files and serve as a file server to distribute the sliced files to other worker nodes.

**Step 1:** To launch the master node, run the following command on laptop 1:
```commandline
# Run the master node on laptop 1 (IP: 192.168.2.1, RANK = 0)
> python examples/run_multihost.py --rank 0 --world_size 2 --master_ip 192.168.2.1 --master_port=29500 --model_type llama --model_path /root/TPI-LLM/pretrained_models/Llama-2-1.1b-ft --prompt "how are you?" --length 20 --memory_window 4
```

> **NOTE:** Please make sure the master node can be connected by all other nodes. The master node also participate in tensor-parallel inference.

**Step 2**: To launch other worker nodes, use the following command on other laptops (e.g., laptop 2):
```commandline
# Run the worker node on host 2 (IP: 192.168.2.2, RANK = 1)
> python examples/run_multihost.py --rank 1 --world_size 2 --master_ip 192.168.2.1 --master_port=29500 --model_type llama --model_path /root/TPI-LLM/pretrained_models/sync --memory_window 4
```

The worker nodes will automatically download their weight files from the master node. If you have downloaded
the files before, you can use the option `--force_download` to force a re-download.

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

# Cite Us
Upcoming, the paper is under review.