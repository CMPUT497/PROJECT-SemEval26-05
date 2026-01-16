import json

input_path = "predictions/best_score_prev.jsonl"
ids = []
with open(input_path, "r", encoding="utf8") as f:
    data = f.readlines()
    for line in data:
        line = json.loads(line)
        pred = line["prediction"]
        if pred == 2:
            ids.append(line["id"])
print(ids)
print(len(ids))

file_input = "data/dev.json"
new_file = "data/new_dev.json"
new_data = {}
with open(file_input, "r", encoding="utf8") as f:
    data = json.load(f)
    for i, id in enumerate(ids):
        new_data[id] = data[id]
        
with open(new_file, "w", encoding="utf8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)