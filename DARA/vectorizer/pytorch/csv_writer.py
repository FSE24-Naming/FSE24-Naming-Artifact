import csv
import json

def flatten_dict(dd, separator='_', prefix=''):
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }

# Load your JSON data:
with open('././ptm_vectors/failed_models_reason_subset_detailed.json') as json_file:
    data = json.load(json_file)

data = flatten_dict(data)

# Now, let's write the data to a CSV file
with open('././ptm_vectors/failed_models_reason_subset_detailed.csv', 'w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=data.keys())
    writer.writeheader()
    writer.writerow(data)



