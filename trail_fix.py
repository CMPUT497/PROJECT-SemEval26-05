import json

input_path = "predictions/method3_v2_predictions.jsonl"
results = []
with open(input_path, "r", encoding="utf8") as f:
    data = f.readlines()
    for line in data:
        line = json.loads(line)
        ids = line["id"]
        pred = line["prediction"]
        for id in ids:
            output = {"id": id, "prediction": pred}
            results.append(output)

output_path = "predictions/method3_v2_predictions.jsonl"
with open(output_path, "w", encoding="utf8") as fout:
    for item in results:
        fout.write(json.dumps(item) + "\n")
