import json

with open('././ptm_vectors/vec_p.json') as f:
    data = json.load(f)

k_set = set()
for k, v in data.items():
    for kk, vv in v.items():
        for kkk, vvv in vv.items():
            k_set.add(kkk)

print(len(k_set))