from typing import Literal, List
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score
from fairlearn.metrics import equalized_odds_difference, equalized_odds_ratio
from itertools import combinations


def accuracy(gt_answer_list, pred_answer_list, sensitive_features: List = None):
    if sensitive_features is None:
        correct = 0
        for gt_answer, pred_answer in zip(gt_answer_list, pred_answer_list):
            if gt_answer == pred_answer:
                correct += 1
        return correct / len(gt_answer_list)
    else:
        # Return accuracy for each value of the sensitive attribute
        # E.g., returns {0: 0.9, 2: 0.98, 100: 0.99}
        available_values = set(sensitive_features)
        results = {}
        for value in available_values:
            correct = 0
            total = 0
            for gt_answer, pred_answer, sensitive_feature in zip(gt_answer_list, pred_answer_list, sensitive_features):
                if sensitive_feature == value:
                    total += 1
                    if gt_answer == pred_answer:
                        correct += 1
            results[value] = float(correct / total)
        return results


def _ece(gt_answer_list, pred_answer_list, conf_list, uniform=True, bins=10, verbose=False):
    """
    Compute Expected Calibration Error (ECE) with support for uniform and quantile binning (Q-ECE).

    Args:
        gt_answer_list (list or np.array): Ground truth labels (0 or 1).
        pred_answer_list (list or np.array): Predicted labels (0 or 1).
        conf_list (list or np.array): Model confidence scores
        uniform (bool): If True, use uniform binning (ECE), otherwise use quantile binning (Q-ECE).
        bins (int): Number of bins.
        verbose (bool): If True, prints detailed bin statistics in table format.

    Returns:
        float: ECE or Q-ECE value.
    """
    gt_answer_list = np.array(gt_answer_list)
    pred_answer_list = np.array(pred_answer_list)
    conf_list = np.array(conf_list)

    if uniform:
        # Uniform binning (ECE)
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_indices = np.digitize(conf_list, bin_edges, right=True) - 1  # Assign bins
    else:
        # Quantile binning (Q-ECE)
        sorted_indices = np.argsort(conf_list)
        sorted_conf = conf_list[sorted_indices]
        sorted_gt = gt_answer_list[sorted_indices]
        bin_size = len(conf_list) // bins
        bin_indices = np.zeros_like(conf_list, dtype=int)

        for i in range(bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i != bins - 1 else len(conf_list)
            bin_indices[sorted_indices[start_idx:end_idx]] = i

    ece = 0.0
    bin_data = []

    for i in range(bins):
        bin_mask = (bin_indices == i)
        if not np.any(bin_mask):
            continue

        bin_conf = conf_list[bin_mask]
        bin_gt = gt_answer_list[bin_mask]
        bin_pred = pred_answer_list[bin_mask]

        avg_conf = np.mean(bin_conf)
        avg_acc = np.mean(bin_gt == bin_pred)
        bin_size = len(bin_conf)

        ece += (abs(avg_acc - avg_conf) * bin_size)

        bin_range = (
            (bin_edges[i], bin_edges[i + 1]) if uniform else (bin_conf.min(), bin_conf.max())
        )
        bin_data.append([i, bin_range[0], bin_range[1], bin_size, avg_conf, avg_acc])

    # Print table if verbose
    if verbose:
        print("\n" + ("-" * 65))
        print(f"{'Bin':<5} {'Range':<12} {'Count':<8} {'Avg Conf':<12} {'Avg Acc':<12}")
        print("-" * 65)
        for row in bin_data:
            print(f"{row[0]:<5} ({row[1]:.2f}, {row[2]:.2f})  {row[3]:<8} {row[4]:.4f}      {row[5]:.4f}")
        print("-" * 65)

    return float(ece / len(conf_list))


def ece(gt_answer_list, pred_answer_list, conf_list, sensitive_features: List = None, uniform=True, bins=10, verbose=False):
    if sensitive_features is None:
        return _ece(gt_answer_list, pred_answer_list, conf_list, uniform=uniform, bins=bins, verbose=verbose)
    else:
        # Return ECE for each value of the sensitive attribute
        # E.g., returns {0: 0.1, 2: 0.05, 100: 0.02}
        available_values = set(sensitive_features)
        results = {}
        for value in available_values:
            selected_gt_answer_list = [gt_answer for gt_answer, sensitive_feature in zip(gt_answer_list, sensitive_features) if sensitive_feature == value]
            selected_pred_answer_list = [pred_answer for pred_answer, sensitive_feature in zip(pred_answer_list, sensitive_features) if sensitive_feature == value]
            selected_conf_list = [conf for conf, sensitive_feature in zip(conf_list, sensitive_features) if sensitive_feature == value]
            results[value] = _ece(selected_gt_answer_list, selected_pred_answer_list, selected_conf_list, uniform=uniform, bins=bins, verbose=verbose)
        return results


def eod(gt_answer_list, pred_answer_list, sensitive_features_dict: dict, agg: Literal["worst_case", "mean"] = "mean"):
    max_vals = {key: max(rv) + 1 for key, rv in sensitive_features_dict.items()}  # Compute max values for encoding
    joint_rv = []
    for values in zip(*sensitive_features_dict.values()):
        encoded_value = 0
        factor = 1
        for key in sensitive_features_dict:
            encoded_value += values[list(sensitive_features_dict.keys()).index(key)] * factor
            factor *= max_vals[key]
        joint_rv.append(encoded_value)
    results = {}
    for key in sensitive_features_dict:
        results["eod_" + key] = float(equalized_odds_difference(gt_answer_list, pred_answer_list,
                                                                sensitive_features=sensitive_features_dict[key], agg=agg))
    results["eod_joint"] = float(equalized_odds_difference(gt_answer_list, pred_answer_list,
                                                           sensitive_features=joint_rv, agg=agg))
    return results


def eor(gt_answer_list, pred_answer_list, sensitive_features_dict: dict, agg: Literal["worst_case", "mean"] = "mean"):
    max_vals = {key: max(rv) + 1 for key, rv in sensitive_features_dict.items()}  # Compute max values for encoding
    joint_rv = []
    for values in zip(*sensitive_features_dict.values()):
        encoded_value = 0
        factor = 1
        for key in sensitive_features_dict:
            encoded_value += values[list(sensitive_features_dict.keys()).index(key)] * factor
            factor *= max_vals[key]
        joint_rv.append(encoded_value)
    results = {}
    for key in sensitive_features_dict:
        results["eor_" + key] = float(equalized_odds_ratio(gt_answer_list, pred_answer_list,
                                                           sensitive_features=sensitive_features_dict[key], agg=agg))
    results["eor_joint"] = float(equalized_odds_ratio(gt_answer_list, pred_answer_list,
                                                      sensitive_features=joint_rv, agg=agg))
    return results


def esacc(gt_answer_list, pred_answer_list, sensitive_features_dict: dict):
    max_vals = {key: max(rv) + 1 for key, rv in sensitive_features_dict.items()}  # Compute max values for encoding
    joint_rv = []
    for values in zip(*sensitive_features_dict.values()):
        encoded_value = 0
        factor = 1
        for key in sensitive_features_dict:
            encoded_value += values[list(sensitive_features_dict.keys()).index(key)] * factor
            factor *= max_vals[key]
        joint_rv.append(encoded_value)

    acc_bar = accuracy(gt_answer_list, pred_answer_list)
    results = {}
    for key in sensitive_features_dict:
        accs_dict = accuracy(gt_answer_list, pred_answer_list, sensitive_features_dict[key])
        results["esacc_" + key] = float(acc_bar / (1 + np.std(list(accs_dict.values()))))
    accs_dict = accuracy(gt_answer_list, pred_answer_list, joint_rv)
    results["esacc_joint"] = float(acc_bar / (1 + np.std(list(accs_dict.values()))))
    return results


# Please avoid using non-binary sensitive attributes, because the combination of non-binary sensitive attributes will be too large
def eceg(gt_answer_list, pred_answer_list, conf_list, sensitive_features_dict: dict, uniform=True, bins=10, agg: Literal["worst_case", "mean"] = "mean"):
    if agg != "mean":
        raise ValueError("ECE gap only supports mean aggregation.")
    max_vals = {key: max(rv) + 1 for key, rv in sensitive_features_dict.items()}  # Compute max values for encoding
    joint_rv = []
    for values in zip(*sensitive_features_dict.values()):
        encoded_value = 0
        factor = 1
        for key in sensitive_features_dict:
            encoded_value += values[list(sensitive_features_dict.keys()).index(key)] * factor
            factor *= max_vals[key]
        joint_rv.append(encoded_value)
    # Mean ECE gap for each sensitive attribute
    eceg_dict = {}
    for key in sensitive_features_dict:
        ece_dict = ece(gt_answer_list, pred_answer_list, conf_list, sensitive_features=sensitive_features_dict[key], uniform=uniform, bins=bins)
        # Calculate the mean ECE gap
        # Extract values
        values = list(ece_dict.values())
        # Compute mean difference
        differences = [abs(x - y) for x, y in combinations(values, 2)]
        mean_difference = float(sum(differences) / len(differences))
        eceg_dict["eceg_" + key] = mean_difference
    # Mean ECE gap for the joint sensitive attribute
    ece_dict = ece(gt_answer_list, pred_answer_list, conf_list, sensitive_features=joint_rv, uniform=uniform, bins=bins)
    # Calculate the mean ECE gap
    # Extract values
    values = list(ece_dict.values())
    # Compute mean difference
    differences = [abs(x - y) for x, y in combinations(values, 2)]
    mean_difference = float(sum(differences) / len(differences))
    eceg_dict["eceg_joint"] = mean_difference
    return eceg_dict


def esece(gt_answer_list, pred_answer_list, conf_list, sensitive_features_dict: dict, uniform=True, bins=10):
    max_vals = {key: max(rv) + 1 for key, rv in sensitive_features_dict.items()}  # Compute max values for encoding
    joint_rv = []
    for values in zip(*sensitive_features_dict.values()):
        encoded_value = 0
        factor = 1
        for key in sensitive_features_dict:
            encoded_value += values[list(sensitive_features_dict.keys()).index(key)] * factor
            factor *= max_vals[key]
        joint_rv.append(encoded_value)
    ece_bar = _ece(gt_answer_list, pred_answer_list, conf_list, uniform=uniform, bins=bins, verbose=False)
    results = {}
    for key in sensitive_features_dict:
        eces_dict = ece(gt_answer_list, pred_answer_list, conf_list, sensitive_features=sensitive_features_dict[key], uniform=uniform, bins=bins)
        results["esece_" + key] = float(ece_bar / (1 - np.std(list(eces_dict.values()))))
    eces_dict = ece(gt_answer_list, pred_answer_list, conf_list, sensitive_features=joint_rv, uniform=uniform, bins=bins)
    results["esece_joint"] = float(ece_bar / (1 - np.std(list(eces_dict.values()))))
    return results


def find_best_worst_acc_case(gt_answer_list, pred_answer_list, sensitive_features_dict: dict):
    """
    Find the best and worst case scenarios for accuracy across combinations of sensitive attributes.

    Args:
        gt_answer_list: List of ground truth labels, e.g., [1, 1, 1, 1]
        pred_answer_list: List of predicted labels, e.g., [0, 1, 0, 1]
        sensitive_features_dict: Dictionary of sensitive attributes, e.g., {"sex": [1, 0, 1, 0], "age": [0, 0, 1, 1]}

    Returns:
        Dictionary containing best and worst case combinations of sensitive attributes
    """
    # Get unique values for each sensitive attribute
    unique_values = {
        attr: list(set(values))
        for attr, values in sensitive_features_dict.items()
    }

    # Initialize best and worst cases
    best_case = {}
    worst_case = {}
    best_accuracy = -1
    worst_accuracy = float('inf')

    # Generate all possible combinations of attribute values
    def generate_combinations(attrs, current_combo):
        if not attrs:
            # Calculate accuracy for this combination
            correct = 0
            total = 0

            # Check each data point
            for i in range(len(gt_answer_list)):
                # Check if this point matches all attribute values in current combination
                matches_combo = all(
                    sensitive_features_dict[attr][i] == current_combo[attr]
                    for attr in current_combo
                )

                if matches_combo:
                    total += 1
                    if gt_answer_list[i] == pred_answer_list[i]:
                        correct += 1

            # Calculate accuracy if we have any matching points
            if total > 0:
                accuracy = correct / total

                nonlocal best_accuracy, worst_accuracy, best_case, worst_case

                # Update best case if accuracy is higher
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_case = current_combo.copy()

                # Update worst case if accuracy is lower
                if accuracy < worst_accuracy:
                    worst_accuracy = accuracy
                    worst_case = current_combo.copy()

            return

        # Get next attribute to process
        current_attr = attrs[0]
        remaining_attrs = attrs[1:]

        # Try each possible value for this attribute
        for value in unique_values[current_attr]:
            current_combo[current_attr] = value
            generate_combinations(remaining_attrs, current_combo)

    # Start generating combinations
    generate_combinations(list(sensitive_features_dict.keys()), {})

    # Return results
    return {
        "best_acc_case": {
            "combination": best_case,
            "accuracy": float(best_accuracy)},
        "worst_acc_case": {
            "combination": worst_case,
            "accuracy": float(worst_accuracy)}
    }


def find_best_worst_ece_case(gt_answer_list, pred_answer_list, conf_list, sensitive_features_dict: dict, uniform=True,
                             bins=10):
    """
    Find the best and worst case scenarios for Expected Calibration Error (ECE) across combinations of sensitive attributes.
    Best case is when ECE is lowest (closer to perfect calibration), worst case is when ECE is highest.

    Args:
        gt_answer_list: List of ground truth labels, e.g., [1, 1, 1, 1]
        pred_answer_list: List of predicted labels, e.g., [0, 1, 0, 1]
        conf_list: List of confidence scores, e.g., [0.7, 0.8, 0.6, 0.9]
        sensitive_features_dict: Dictionary of sensitive attributes, e.g., {"sex": [1, 0, 1, 0], "age": [0, 0, 1, 1]}
        uniform: Boolean indicating whether to use uniform binning (True) or quantile binning (False)
        bins: Number of bins to use for ECE calculation

    Returns:
        Dictionary containing best and worst case combinations of sensitive attributes
    """
    # Get unique values for each sensitive attribute
    unique_values = {
        attr: list(set(values))
        for attr, values in sensitive_features_dict.items()
    }

    # Initialize best and worst cases
    best_case = {}
    worst_case = {}
    best_ece = float('inf')  # Lower ECE is better
    worst_ece = -1  # Higher ECE is worse

    # Generate all possible combinations of attribute values
    def generate_combinations(attrs, current_combo):
        if not attrs:
            # Prepare filtered lists for this combination
            indices = range(len(gt_answer_list))
            matching_indices = [
                i for i in indices
                if all(sensitive_features_dict[attr][i] == current_combo[attr]
                       for attr in current_combo)
            ]

            # Only proceed if we have enough data points for meaningful ECE calculation
            if len(matching_indices) >= bins:
                # Filter data for matching indices
                filtered_gt = [gt_answer_list[i] for i in matching_indices]
                filtered_pred = [pred_answer_list[i] for i in matching_indices]
                filtered_conf = [conf_list[i] for i in matching_indices]

                # Calculate ECE for this combination
                current_ece = _ece(
                    filtered_gt,
                    filtered_pred,
                    filtered_conf,
                    uniform=uniform,
                    bins=bins,
                    verbose=False
                )

                nonlocal best_ece, worst_ece, best_case, worst_case

                # Update best case if ECE is lower
                if current_ece < best_ece:
                    best_ece = current_ece
                    best_case = current_combo.copy()

                # Update worst case if ECE is higher
                if current_ece > worst_ece:
                    worst_ece = current_ece
                    worst_case = current_combo.copy()

            return

        # Get next attribute to process
        current_attr = attrs[0]
        remaining_attrs = attrs[1:]

        # Try each possible value for this attribute
        for value in unique_values[current_attr]:
            current_combo[current_attr] = value
            generate_combinations(remaining_attrs, current_combo)

    # Start generating combinations
    generate_combinations(list(sensitive_features_dict.keys()), {})

    # Return results with ECE values
    return {
        "best_ece_case": {
            "combination": best_case,
            "ece_value": float(best_ece)
        },
        "worst_ece_case": {
            "combination": worst_case,
            "ece_value": float(worst_ece)
        }
    }
