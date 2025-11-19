import json

input_path = "input/ref/solution.jsonl"
output_path = "input/ref/solutions_avg.JSONL"

with open(input_path, "r", encoding="utf8") as fin, open(output_path, "w", encoding="utf8") as fout:
    for line in fin:
        data = json.loads(line)
        id_ = data["id"]
        labels = data["label"]
        avg_score = sum(labels) / len(labels) if labels else 0
        output = {"id": id_, "avg_score": avg_score}
        fout.write(json.dumps(output) + "\n")
