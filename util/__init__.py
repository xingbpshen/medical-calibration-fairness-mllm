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


def user_warning(msg):
    warnings.warn(msg, UserWarning)


def info(file_name, msg):
    print(f"\033[1;94m[{file_name}]\033[0m \033[94mINFO\033[0m {msg}")


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


def is_ready(port):
    """
    Check if the vllm server is ready to accept requests.
    :param port: The port on which the vllm server is running.
    :return: True if the server is ready, False otherwise.
    """
    url = f"http://localhost:{port}/health/"
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException as exc:
        return False
    else:
        if response.status_code == 200:
            return True


def wait_until_ready(port, subproc, timeout=1800):
    """
    Wait until the vllm server is ready to accept requests.
    :param port: The port on which the vllm server is running.
    :param subproc: The subprocess object for the vllm server.
    :param timeout: The maximum time to wait in seconds.
    """
    start_time = time.time()
    while not is_ready(port):
        # if the server has exited, raise an error
        if subproc.poll() is not None:
            stderr_output = subproc.stderr.read().decode()
            raise RuntimeError(
                f"Error:\n{stderr_output}\nvLLM server at port {port} exited unexpectedly, please kill the corresponding GPU process manually by:\nkill -9 PID")
        if time.time() - start_time > 30:
            info('util.__init__.py', 'Still waiting? Check the GPU mem usage to make sure no server is lost.')
        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Server at port {port} did not become ready within {timeout} seconds.")
        time.sleep(10)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
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
