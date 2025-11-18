import json

input_path = "input/ref/solution.jsonl"

with open(input_path, "r", encoding="utf8") as f:
    for line in f:
        data = json.loads(line)
        id_ = data["id"]
        labels = data["label"]
        avg_score = sum(labels) / len(labels) if labels else 0
        print(f"{id_}\t{avg_score}")
