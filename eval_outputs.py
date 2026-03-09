import argparse
import json
import os, sys
from typing import Optional

import torch
import numpy as np
from transformers import AutoTokenizer

from ttrl.verifier.auto_verify import auto_verify
import data

import pdb

# Default prompts
BASE_MODEL_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n<|user|>\n{}\n<|assistant|>\n<think>"

def evaluate(response_json, dataset_name=None, filter_errors=False):

    assert isinstance(response_json, list) and len(response_json) > 0, "`response_json` must be a non-empty list"

    response_json = sorted(response_json, key=lambda x: x["global_index"])

    def get_outputs(response):
        outputs = response.get("responses")
        if isinstance(outputs, list):
            return outputs
        raise TypeError("response['responses'] must be a list.")

    rollouts = len(get_outputs(response_json[0]))

    if filter_errors:
        def filter_criterion(response):
            outputs = get_outputs(response)
            all_errors = ['OutOfMemoryError', 'ValueError', 'RuntimeError', 'Exception']
            for output in outputs:
                if any([output.startswith(error) for error in all_errors]):
                    return False
                if len(output) == 0:
                    return False
                    
            return True

        response_json = [response for response in response_json if filter_criterion(response)]

    dict_accuracies = {
        f"avg@{rollouts}": 0.0,
        f"pass@{rollouts}": 0.0,
        f"metadata": response_json.copy(),
    }

    all_labels = []
    if isinstance(response_json[0], dict) and "label" in response_json[0]:
        for response in response_json:
            all_labels.append(response["label"])
    elif dataset_name is not None:
        all_labels = data.read_labels_from_benchmark(dataset_name)
    else:
        raise ValueError("Cannot determine labels from `response_json` or `dataset_name`")

    all_outputs = []
    for response in response_json:
        outputs = get_outputs(response)
        all_outputs.append(outputs)

        if rollouts != len(outputs):
            print(f"Detect rollouts mismatch: {rollouts} != {len(outputs)}")

    all_outputs = list(zip(*all_outputs))

    # Calculate accuracies
    verify_task = "math"
    all_accuracies = [auto_verify(verify_task, 1, outputs, all_labels) for outputs in all_outputs]
    accuracy_at_k = np.mean([np.mean(acc) for acc in all_accuracies])
    temp_all_accuracies = np.array(all_accuracies)
    pass_at_k = temp_all_accuracies.max(axis=0).mean()

    # Update metadata
    metadata = dict_accuracies["metadata"]
    for idx, response in enumerate(metadata):
        response["rewards"] = [a[idx] for a in all_accuracies]

    dict_accuracies[f"avg@{rollouts}"] = accuracy_at_k
    dict_accuracies[f"pass@{rollouts}"] = pass_at_k

    return dict_accuracies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to output file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file")
    parser.add_argument("--filter_errors", action="store_true",
                        help="Filter out responses that contain errors")

    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        response_json = json.load(f)

    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    dict_accuracies = evaluate(response_json, filter_errors=args.filter_errors)

    with open(args.output_file, "w") as f:
        json.dump(dict_accuracies, f, indent=4)
