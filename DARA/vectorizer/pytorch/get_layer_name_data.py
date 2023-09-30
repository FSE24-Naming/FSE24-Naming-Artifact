import json, os
    
def read_json(filepath):
    with open(filepath) as f:
        return json.load(f)

def write_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f)

data_folder = './vectorizer/pytorch/ptm_data/'

layer_name_data = []

file_names = os.listdir(data_folder)
cnt = 0

for fn in file_names:

    replaced_fn = fn.replace('>', '/')
    
    data = read_json(data_folder + fn)
    layer_name_data.append(' '.join([layer['operation'] for layer in data]))

    cnt += 1

    print(f'[{cnt}] Finished {replaced_fn}')

write_json(layer_name_data, './vectorizer/pytorch/layer_name_data.json')