import random
import json
import os

with open('./model_collection/filtered_models.json') as f:
    data = json.load(f)

all_m = set()
for k, v in data.items():
    for n in v:
        all_m.add(n)

def slash_change(s):
    ns = ''
    for c in s:
        if c == '>':
            nc = '/'
        else:
            nc = c
        ns += nc
    return ns

ld = os.listdir('././ptm_data')
nld = [slash_change(s)[:-5] for s in ld]

for n in nld:
    if n in all_m:
        all_m.remove(n)

MANUAL_SKIP = {
        "mrm8488/convbert-small-spanish",
        "mrm8488/convbert-base-spanish",
        "inverse-scaling/opt-30b_eval",
        "google/switch-xxl-128",
        "google/switch-large-128",
        "bigscience/bloomz",
        "bigscience/bloomz-560m",
        "bigscience/bloomz-1b1",
        "bigscience/bloomz-p3",
        "bigscience/bloomz-7b1",
        "bigscience-data/sgpt-bloom-1b7-nli",
        "bs-la/bloomz-7b1-4b-ru",
        "bigscience/bloomz-7b1-mt",
        "bigscience/bloom-3b",
        "bigscience/bloomz-3b",
        "bigscience/bloomz-mt",
        "bigscience/bloom",
        "model-attribution-challenge/bloom-2b5",
        "bigscience/bloomz-1b7",
        "bigscience/sgpt-bloom-7b1-msmarco",
        "bigscience/bloomz-7b1-p3",
        "AlekseyKorshuk/amazon-reviews-input-output-6.7b-best",
        "AlekseyKorshuk/6.7b-dalio-book-handwritten-io-constant-1e-6-v2"
}
for n in MANUAL_SKIP:
    if n in all_m:
        all_m.remove(n)

ua_m = list(all_m)

selected_models = random.choices(population=ua_m, k=50)

with open('././ptm_vectors/failed_models_subset_2.json', 'w') as f:
    json.dump(selected_models, f)
