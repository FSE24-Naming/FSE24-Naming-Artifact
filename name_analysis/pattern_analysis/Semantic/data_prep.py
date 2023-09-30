import json, os

with open('filtered_models.json') as f:
    filtered_models = json.load(f)

name_arch_map = dict()
sim_name_map = dict()
name_order = list()
for arch, model_name_list in filtered_models.items():
    for model_name in model_name_list:
        name_arch_map[model_name] = arch
        sim_name_map[model_name] = model_name.split('/')[-1]
        name_order.append(model_name)

with open('name_arch_map.json', 'w') as f:
    json.dump(name_arch_map, f)
with open('sim_name_map.json', 'w') as f:
    json.dump(sim_name_map, f)
with open('name_order.json', 'w') as f:
    json.dump(name_order, f)