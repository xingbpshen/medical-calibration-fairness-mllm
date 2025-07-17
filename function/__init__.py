import numpy as np
from typing import List
from mllm import BaseMLLM
from prompt import BasePrompt


def exponential_decay(u: np.ndarray, s_a: np.ndarray, alpha: float) -> np.ndarray:
    assert u.ndim == 2 and s_a.ndim == 2
    assert u.shape == s_a.shape
    # extract the diagonals
    u_diag = np.diag(u)
    s_a_diag = np.diag(s_a)
    # compute the new diagonal values with exponential decay
    c_a = u_diag + (s_a_diag - u_diag) * np.exp(- 1 / np.sqrt(alpha + 1) * np.abs(s_a_diag - u_diag))
    # return a new diagonal matrix
    return np.diag(c_a)


def compute_alpha(u: np.ndarray, s: List[np.ndarray]) -> float:
    # check if all elements in s are 2D arrays
    assert all(isinstance(arr, np.ndarray) and arr.ndim == 2 for arr in s)
    # check if the shape of u matches the shape of all elements in s
    assert all(arr.shape == u.shape for arr in s)
    # extract the diagonals
    u_diag = np.diag(u)
    alpha = 0.0
    for s_a in s:
        s_a_diag = np.diag(s_a)
        # new alpha is the infinity norm of the absolute difference between the diagonals
        new_alpha = np.linalg.norm(np.abs(s_a_diag - u_diag), ord=np.inf)
        alpha = max(alpha, new_alpha)
    return alpha


def prob_dict_to_vector(prob_dict: dict) -> np.ndarray:
    # convert a probability dictionary to a vector
    num_categories = len(prob_dict)
    prob_vector = np.zeros(num_categories)
    # keys in prob_dict can be any, so we follow the existing order of the keys
    for idx, key in enumerate(prob_dict.keys()):
        prob_vector[idx] = prob_dict[key]
    return prob_vector


def softmax(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 1
    e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return e_x / e_x.sum()


def query_calibrated_answer_probs(data_point: dict, use_prompt: BasePrompt, use_mllm: BaseMLLM, cared_attribute: str, **kwargs) -> (int, dict):
    """
    Query the calibrated answer probabilities for a given data point using the provided prompt and MLLM.

    Args:
        data_point (dict): The data point containing the image path and attributes.
        use_prompt (BasePrompt): The prompt to format the query.
        use_mllm (BaseMLLM): The MLLM instance to use for querying.
        cared_attribute (str): An attribute to consider fairness.
        **kwargs: Additional keyword arguments for the MLLM query, e.g., temperature, max_completion_tokens.

    Returns:
        int: The predicted answer (we assume that the label is in integers).
        dict: A dictionary mapping answer indices to their calibrated probabilities.
    """
    # get the un-calibrated answer probabilities
    baseline_prompt = use_prompt.build(data_point=data_point, marginalize_out=[])
    _, baseline_prob_dict = use_prompt.parse_response(use_mllm.chat_completion(baseline_prompt, **kwargs))

    # compute calibration parameters
    # population-level (l1)
    # may causes redundant computation if the total attributes of the datapoint are just one cared attribute
    l1_prompt = use_prompt.build(data_point=data_point, marginalize_out=['image', cared_attribute])
    _, l1_prob_dict = use_prompt.parse_response(use_mllm.chat_completion(l1_prompt, **kwargs))
    u = np.linalg.inv(np.diag(prob_dict_to_vector(l1_prob_dict)))
    # subgroup-level (l2)
    # may causes redundant computation if the total attributes of the datapoint are just one cared attribute
    s_in_dict = {}
    for a in getattr(use_prompt.attributes_namespace, cared_attribute):
        _data_point = data_point.copy()
        _data_point[cared_attribute] = a
        l2_prompt = use_prompt.build(data_point=_data_point, marginalize_out=['image'])
        _, l2_prob_dict = use_prompt.parse_response(use_mllm.chat_completion(l2_prompt, **kwargs))
        s_in_dict[a] = np.linalg.inv(np.diag(prob_dict_to_vector(l2_prob_dict)))
    # compute c_a
    alpha = compute_alpha(u, list(s_in_dict.values()))
    c_a = exponential_decay(u, s_in_dict[data_point[cared_attribute]], alpha)
    # compute calibrated probabilities
    calibrated_probs = softmax(np.dot(c_a, prob_dict_to_vector(baseline_prob_dict)))
    calibrated_prob_dict = dict(zip(baseline_prob_dict.keys(), calibrated_probs))
    return max(calibrated_prob_dict, key=calibrated_prob_dict.get), calibrated_prob_dict
