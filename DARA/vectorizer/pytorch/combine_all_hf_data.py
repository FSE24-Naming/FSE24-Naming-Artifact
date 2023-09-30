import json, os, pickle
    
def read_json(filepath):
    with open(filepath) as f:
        return json.load(f)

def write_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f)

one_skip_two_gram_layer_filepath = './vectorizer/pytorch/ptm_oneskip_twogram_layer'
one_skip_two_gram_layerdim_filepath = './vectorizer/pytorch/ptm_oneskip_twogram_layerdim'
data_folder = './vectorizer/pytorch/ptm_data/'

# Read files once
#two_gram_data = read_json(two_gram_filepath)
#one_skip_two_gram_data = read_json(one_skip_two_gram_filepath)
two_gram_data = dict()
one_skip_two_gram_data = dict()

file_names = os.listdir(data_folder)
cnt = 0

for fn in file_names:
    replaced_fn = fn.replace('>', '/')
    if replaced_fn in two_gram_data:
        print(f'[{cnt}] Skipped {replaced_fn}')
        continue
    
    data = read_json(data_folder + fn)
    new_abs_model = [{k:v for k,v in layer.items() if k not in ['node_id', 'connects_to']} for layer in data]
    

    one_skip_two_gram_layer = [new_abs_model[i]['operation'] + '|' + new_abs_model[i+1]['operation'] for i in range(len(new_abs_model) - 1)]
    one_skip_two_gram_layer_dim = [[str(new_abs_model[i]) + '|' + str(new_abs_model[i+1]), str(new_abs_model[i]) + '|' + str(new_abs_model[i+2])] for i in range(len(new_abs_model) - 2)]
    
    #two_gram_data[replaced_fn] = two_gram
    #one_skip_two_gram_data[replaced_fn] = one_skip_two_gram

    cnt += 1
    

    #if cnt == 2:
    #    break

    # Write files once
    write_json(one_skip_two_gram_layer, one_skip_two_gram_layer_filepath + '/' + fn)
    write_json(one_skip_two_gram_layer_dim, one_skip_two_gram_layerdim_filepath + '/' + fn)

    print(f'[{cnt}] Finished {replaced_fn}')
