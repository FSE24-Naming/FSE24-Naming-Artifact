from data_cleaner import read_from_json, write_to_json

emr = read_from_json('clusters.json')
ol = read_from_json('outliers.json')

model_list = []
for k, v in emr.items():
    for kk, vv in v.items():
        for ll in vv:
            model_list.append(ll)

for k, v in ol.items():
    for ll in v:
        model_list.append(ll)

write_to_json(model_list, 'external_model_list.json')