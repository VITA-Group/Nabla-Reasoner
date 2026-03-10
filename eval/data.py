import os
import json

BENCHMARKS = {
    "MATH-500": {
        "path": "data/MATH-TTT/test.json",
        "rollouts": 1,
    },
    "AIME-2024": {
        "path": "data/AIME-TTT/test.json",
        "rollouts": 8,
    },
    "AIME-2025": {
        "path": "data/AIME2025-TTT/test.json",
        "rollouts": 8,
    },
    "AMC": {
        "path": "data/AMC-TTT/test.json",
        "rollouts": 8,
    },
}

def available_datasets():
    return list(BENCHMARKS.keys())

def load_benchmark_dataset(dataset_name):
    dataset_info = BENCHMARKS[dataset_name]
    test_path = dataset_info["path"]

    # Load test data
    with open(test_path) as f:
        test_data = json.load(f)

    return test_data

def read_prompts_from_benchmark(dataset_name):
    test_data = load_benchmark_dataset(dataset_name)

    return [x["prompt"] for x in test_data]

def read_labels_from_benchmark(dataset_name):
    test_data = load_benchmark_dataset(dataset_name)

    return [x["answer"] for x in test_data]

def read_prompts_from_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompts file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    prompts = []

    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    prompts.append(s)

    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "prompts" in data and isinstance(data["prompts"], list):
            prompts = [str(p) for p in data["prompts"]]
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                prompts = [str(p["prompt"]) for p in data]
            else:
                prompts = [str(p) for p in data]
        else:
            raise ValueError("Unsupported JSON format. Use {'prompts': [...]} or a raw list.")

    elif ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "prompt" in obj:
                        prompts.append(str(obj["prompt"]))
                    elif isinstance(obj, str):
                        prompts.append(obj)
                    else:
                        raise ValueError
                except Exception:
                    # Treat as raw string line if not valid JSON
                    prompts.append(line)
    else:
        raise ValueError("Unsupported prompts file extension. Use .txt, .json, or .jsonl")

    return prompts