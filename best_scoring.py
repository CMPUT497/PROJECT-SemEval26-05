import json

input_file = 'predictions/best_score_floating_point.jsonl'
output_file = 'predictions/best_score_int.json'

results = []
with open(input_file, 'r') as fin:
    for line in fin:
        entry = json.loads(line)
        if entry['prediction'] - int(entry['prediction']) < 0.4:
            entry['prediction'] = int(entry['prediction'])
        else:
            entry['prediction'] = int(entry['prediction']) + 1
        results.append(entry)

with open(output_file, 'w') as fout:
    for entry in results:
        fout.write(json.dumps(entry) + "\n")
