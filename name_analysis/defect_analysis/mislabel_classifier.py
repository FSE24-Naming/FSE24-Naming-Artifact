from data_cleaner import write_to_json, read_from_json, find_mismatch
import re
from loguru import logger

'''
name_arch_map = read_from_json('name_arch_map.json')
name_simarch_map = dict()
for k, v in name_arch_map.items():
    name_simarch_map[k] = re.split('For|Model|LMHead|Small|Base|v1|v2|V1|V2', v)[0]

write_to_json(name_simarch_map, 'name_simarch_map.json')
'''

name_simarch_map = read_from_json('name_simarch_map.json')
combined_output = read_from_json('combined_output.json')
mismatched_models = find_mismatch(combined_output)

model_name_A_set_map = dict()
cnt = 0
for model_name, name_cat_conf_pairs_dict in combined_output.items():
    if model_name in mismatched_models: continue
    A_set = set()
    for seg, cat_conf_pairs in name_cat_conf_pairs_dict.items():
        if cat_conf_pairs == None:
            cnt += 1 
            logger.warning(f'Model name {model_name} ommitted')
            continue
        top1_cat_conf_pair =  cat_conf_pairs[0]
        if top1_cat_conf_pair == None: continue
        cat = top1_cat_conf_pair[0]
        conf = top1_cat_conf_pair[1]
        if cat == 'A' and seg != 'model':
            A_set.add(seg.lower())
    model_name_A_set_map[model_name] = A_set

wrong_arch = list()
uninformable = list()

for model_name, A_set in model_name_A_set_map.items():
    
    model_arch = name_simarch_map[model_name].lower()
    if len(A_set) == 0:
        uninformable.append(model_name)
    elif model_arch not in A_set:
        print(model_name, model_arch, A_set)
        wrong_arch.append(model_name)

print(f'Wrong arch: {len(wrong_arch)}, Uninformable: {len(uninformable)}')
print(f'Omitted models: {cnt}')
print(f'Total Model: {len(combined_output)}')

write_to_json(wrong_arch, 'wrong_arch.json')
write_to_json(uninformable, 'uninformable.json')
        
        




