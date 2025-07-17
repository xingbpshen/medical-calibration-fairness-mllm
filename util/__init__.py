import torch
import warnings
import argparse
import base64
import os
import inflect
import random
import numpy as np
import requests
import time
import yaml
from prompt.papila import PapilaPrompt
from mllm.compatible_server import CompatibleServer
from typing import List
import json


def user_warning(msg):
    warnings.warn(msg, UserWarning)


def info(file_name, msg):
    print(f"\033[1;94m[{file_name}]\033[0m \033[94mINFO\033[0m {msg}")


def get_prompt_class(dataset: str):
    """
    Returns the appropriate prompt class based on the dataset name.
    :param dataset: Name of the dataset (e.g., "papila").
    :return: The corresponding prompt class.
    """
    if dataset == "papila":
        return PapilaPrompt
    else:
        raise ValueError(f"Unsupported dataset: {dataset}.")


def get_mllm_class(service: str):
    """
    Returns the appropriate MLLM class based on the service name.
    :param service: Name of the service (e.g., "local_mllm").
    :return: The corresponding MLLM class.
    """
    if service in ["local_mllm", "azure_openai"]:
        return CompatibleServer
    else:
        raise ValueError(f"Unsupported MLLM service: {service}.")


def get_get_image_url_func(service: str):
    """
    Returns the appropriate function to get the image URL based on the service name.
    :param service: Name of the service (e.g., "local_mllm").
    :return: The corresponding function to get the image URL.
    """
    if service == "azure_openai":
        return get_image_url_for_api_mllm
    elif service == "local_mllm":
        return get_image_url_for_local_mllm
    else:
        raise ValueError(f"Unsupported MLLM service: {service}.")


def get_remaining_data_points(log_file_path: str, all_data_points: List[dict]):
    # first check if the log file (json) exists
    if not os.path.exists(log_file_path):
        return all_data_points
    else:
        # read the log json file to list of dicts
        with open(log_file_path, "r") as f:
            log_data = json.load(f)
        # get the list of data id (image_id) that have been processed
        completed_ids = [item['image_id'] for item in log_data]
        # filter the all_data_points to only include those that are not in completed_ids
        remaining_data_points = [item for item in all_data_points if item['image_id'] not in completed_ids]
        return remaining_data_points


def append_log_and_save(log_file_path: str, log_data: dict):
    """
    Append log data to the log file and save it.
    :param log_file_path: Path to the log file.
    :param log_data: Log data to append.
    """
    # check if the log file exists
    if not os.path.exists(log_file_path):
        # if not, create an empty list and write it to the file
        with open(log_file_path, "w") as f:
            json.dump([], f)

    # read existing log data
    with open(log_file_path, "r") as f:
        existing_log_data = json.load(f)

    # append new log data
    existing_log_data.append(log_data)

    # write back to the log file
    with open(log_file_path, "w") as f:
        json.dump(existing_log_data, f, indent=4)


def get_gpu_compute_capability():
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability(0)
    else:
        user_warning("No CUDA-compatible GPU found.")
        return None


def get_vllm_dtype():
    gpu_compute_capability = get_gpu_compute_capability()
    if gpu_compute_capability is not None:
        if gpu_compute_capability[0] >= 8:  # for GPUs with compute capability >= 8.0, can use bfloat16
            return "bfloat16"
        else:  # for older GPUs, must use float16
            return "half"


def is_ready(host, port):
    """
    Check if the vllm server is ready to accept requests.
    :param port: The port on which the vllm server is running.
    :return: True if the server is ready, False otherwise.
    """
    url = f"http://{host}:{port}/health/"
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException as exc:
        return False
    else:
        if response.status_code == 200:
            return True


def wait_until_ready(host, port, subproc, timeout=1800):
    """
    Wait until the vllm server is ready to accept requests.
    :param host: The host on which the vllm server is running.
    :param port: The port on which the vllm server is running.
    :param subproc: The subprocess object for the vllm server.
    :param timeout: The maximum time to wait in seconds.
    """
    start_time = time.time()
    while not is_ready(host, port):
        # if the server has exited, raise an error
        if subproc.poll() is not None:
            stderr_output = subproc.stderr.read().decode()
            raise RuntimeError(
                f"Error:\n{stderr_output}\nvLLM server at http://{host}:{port} exited unexpectedly, please kill the corresponding GPU process manually by:\nkill -9 PID")
        if time.time() - start_time > 30:
            info('util.__init__.py', 'Still waiting? Check the GPU mem usage to make sure no server is lost.')
        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"vLLM server at http://{host}:{port} did not become ready within {timeout} seconds.")
        time.sleep(10)


def dict_to_namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict_to_namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def encode_image(absolute_image_path):
    with open(absolute_image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_url_for_api_mllm(absolute_image_path):
    """
    Returns a data URL for the image at the specified path.
    For API-based MLLMs, this is used to send images as part of the request.
    """
    assert os.path.exists(absolute_image_path)
    image_data = encode_image(absolute_image_path)
    assert image_data, f"Failed to encode image {absolute_image_path}"
    return f"data:image/jpeg;base64,{image_data}"


def get_image_url_for_local_mllm(absolute_image_path):
    """
    Returns a data URL for the image at the specified path.
    For local MLLMs, this is used to send images as part of the request.
    """
    assert os.path.exists(absolute_image_path)
    return f"file://{absolute_image_path}"


def my_number_to_words(number: int):
    """
    Converts a number to its word representation.
    Uses the `inflect` library to handle pluralization and special cases.
    """
    assert isinstance(number, int), "Input must be an integer"
    p = inflect.engine()
    return p.number_to_words(number, andword=",", zero="zero").replace(", ", " ")


# for openai compatible server completion parsing
def parse_content(top_logprobs, options: list) -> [int, dict]:
    low_options = [option.lower() for option in options]
    cate = {}
    for top_logprob in top_logprobs:
        # Use case insensitive matching
        if top_logprob.token.lower() in low_options:
            cate[top_logprob.token.lower()] = top_logprob.logprob
    if len(cate) != len(options):
        print(f"Warning: Not all options are present in the top logprobs: {cate}")
        cate = {option: -0.1 for option in options}
    # Change from logprobs to probabilities then renormalize
    # Step 1: Convert log probabilities to probabilities
    probs = {token: np.exp(lp) for token, lp in cate.items()}
    # Step 2: Calculate the total probability
    total = sum(probs.values())
    # Step 3: Normalize each probability
    normalized_probs = {token: p / total for token, p in probs.items()}
    cate_idx = {}
    for i in range(len(options)):
        cate_idx[i] = normalized_probs[options[i]]
    # Find the maximum value in the dictionary
    max_value = max(cate_idx.values())
    # Gather all keys that have this maximum value
    max_keys = [key for key, value in cate_idx.items() if value == max_value]
    # Randomly choose one of the keys
    pred_answer = random.choice(max_keys)
    # Example return:
    # 1, {0: 0.1, 1: 0.9}
    return pred_answer, cate_idx


def parse_args_and_configs():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Name of the dataset config file in ./config/ (e.g., papila.yml)"
    )
    parser.add_argument(
        "--log", type=str, default="./log", help="Path for saving running related data (e.g., ./log)"
    )
    parser.add_argument(
        "--trial",
        type=str,
        required=False,
        help="A string for documentation purpose. "
             "Will be the name of the folder inside the log folder and the comet trial name.",
    )
    parser.add_argument("--service", type=str, required=True,
                        help="Name of the service to use (e.g., local_mllm)")
    parser.add_argument("--comment", type=str, default="",
                        help="A string for experiment comment")

    args = parser.parse_args()

    # parse config file
    with open(os.path.join("./config", args.config), "r") as f:
        config = yaml.safe_load(f)
    config = dict_to_namespace(config)

    with open("./config/mllm.yml", "r") as f:
        mllm_config = yaml.safe_load(f)
    mllm_config = dict_to_namespace(mllm_config)

    assert hasattr(mllm_config, args.service)

    # add argument for compute capability
    args.vllm_dtype = get_vllm_dtype()
    # add argument for log folder
    args.log_save_folder = os.path.join(args.log, args.trial) if args.trial else args.log

    return args, config, mllm_config
