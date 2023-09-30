import random
import json

with open('././ptm_vectors/failed_models_reason.json') as f:
    data = json.load(f)

LENGTH = len(data)

model_names = list(data.keys())
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
    model_names.remove(n)
selected_models = random.choices(population=model_names, k=50)

selected_model_reason = {n: data[n] for n in selected_models}

with open('././ptm_vectors/failed_models_reason_subset.json', 'w') as f:
    json.dump(selected_model_reason, f)