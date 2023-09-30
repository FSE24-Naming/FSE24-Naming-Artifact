import json
import csv

# Load your JSON data
with open('././ptm_vectors/vec_p.json') as json_file:
    data = json.load(json_file)

new_data = dict()
for k, v in data.items():
    for kk, vv in v.items():
        new_data[kk] = vv

data = new_data

with open('././ptm_vectors/vec_p.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for k, v in data.items():
        writer.writerow([k] + list(v.keys()))
        writer.writerow([k] + list(v.values()))
