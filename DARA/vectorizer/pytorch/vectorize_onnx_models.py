from onnx_vectorize import vectorize
import json, os, pickle

import sys

# Setting the recursion depth to, say, 5000
sys.setrecursionlimit(5000)

onnx_path = '././ptm_vectors/onnx_vec/'
onnx_d_path = onnx_path + 'vec_d.json'
onnx_l_path = onnx_path + 'vec_l.json'
onnx_p_path = onnx_path + 'vec_p.json'

pytorch_vision_path = 'ANONYMIZED_PATH/pytorchhub_onnx_f/'
keras_app_path = 'ANONYMIZED_PATH/kerasapp_onnx_f/'
keras_nlp_path = 'ANONYMIZED_PATH/kerasnlp_onnx_f/'

def read_vec(dp, lp, pp):
    with open(dp) as f:
        dl = json.load(f)
    with open(lp) as f:
        ll = json.load(f)
    with open(pp) as f:
        pl = json.load(f)
    return dl, ll, pl

def merge_vec(dl, ll, pl, path, dkey, name2arch_map):
    fn_lst = os.listdir(path)
    for fn in fn_lst:
        if fn[-5:] != '.onnx': continue
        print(f'merging {fn}')
        comb_path = path + fn
        d, p, l = vectorize(comb_path)
        model_name = fn[:-5]
        if name2arch_map[model_name] not in dl:
            dl[name2arch_map[model_name]] = dict()
            dl[name2arch_map[model_name]][dkey] = dict()
            pl[name2arch_map[model_name]] = dict()
            pl[name2arch_map[model_name]][dkey] = dict()
            ll[name2arch_map[model_name]] = dict()
            ll[name2arch_map[model_name]][dkey] = dict()
        elif dkey not in dl[name2arch_map[model_name]]:
            dl[name2arch_map[model_name]][dkey] = dict()
            pl[name2arch_map[model_name]][dkey] = dict()
            ll[name2arch_map[model_name]][dkey] = dict()
            
        dl[name2arch_map[model_name]][dkey][model_name] = d
        pl[name2arch_map[model_name]][dkey][model_name] = p
        ll[name2arch_map[model_name]][dkey][model_name] = l
    return dl, ll, pl

def write_vec(dl, ll, pl, dp, lp, pp):
    with open(dp, 'w') as f:
        json.dump(dl, f)
    with open(lp, 'w') as f:
        json.dump(ll, f)
    with open(pp, 'w') as f:
        json.dump(pl, f)

def onnxzoo_classify(l, n2a_map):
    content = l['OnnxModelZoo']
    new_l = dict()
    for model_name, vec in content.items():
        arch = n2a_map[model_name]
        if arch not in new_l:
            new_l[arch] = dict()
        if 'OnnxModelZoo' not in new_l[arch]:
            new_l[arch]['OnnxModelZoo'] = dict()
        new_l[arch]['OnnxModelZoo'][model_name] = vec
    return new_l

def convert_to_pickle(path):

    vec_d, vec_l, vec_p = read_vec(path + 'vec_d.json', path + 'vec_l.json', path + 'vec_p.json')
    path_pkl_d, path_pkl_l, path_pkl_p = path + 'vec_d.pickle', path + 'vec_l.pickle', path + 'vec_p.pickle'
    path_pkl_kd, path_pkl_kl, path_pkl_kp = path + 'k_d.pickle', path + 'k_l.pickle', path + 'k_p.pickle'

    def get_key_set(arch_dict):
        key_set = set()
        for arch, model_hubs in arch_dict.items():
            for model_hub_name, model_list in model_hubs.items():
                for model_name, model_vec in model_list.items():
                    for key, value in model_vec.items():
                        key_set.add(key)
        return sorted(list(key_set))

    def create_default_list(ks):
        return [0 for key in ks]
    
    def add_padding(vec, ks):
        k2i_map = dict()
        for i, k in enumerate(ks):
            k2i_map[k] = i
        padded_vec = dict()
        for arch, model_hubs in vec.items():
            new_model_hubs = dict()
            for model_hub_name, model_list in model_hubs.items():
                new_model_list = dict()
                for model_name, model_vec in model_list.items():
                    new_model_vec = create_default_list(ks)
                    for k, v in model_vec.items():
                        new_model_vec[k2i_map[k]] += v
                    new_model_list[model_name] = new_model_vec
                new_model_hubs[model_hub_name] = new_model_list
            padded_vec[arch] = new_model_hubs
        return padded_vec
                    
    
    k_p = get_key_set(vec_p)
    k_d = get_key_set(vec_d)
    k_l = get_key_set(vec_l)
    
    p_vec_l = add_padding(vec_l, k_l)
    p_vec_d = add_padding(vec_d, k_d)
    p_vec_p = add_padding(vec_p, k_p)

    with open(path_pkl_l, 'wb') as f:
        pickle.dump(p_vec_l, f)
    with open(path_pkl_d, 'wb') as f:
        pickle.dump(p_vec_d, f)
    with open(path_pkl_p, 'wb') as f:
        pickle.dump(p_vec_p, f)
    with open(path_pkl_kl, 'wb') as f:
        pickle.dump(k_l, f)
    with open(path_pkl_kd, 'wb') as f:
        pickle.dump(k_d, f)
    with open(path_pkl_kp, 'wb') as f:
        pickle.dump(k_p, f)


'''
with open('././model_hub_arch_class.json') as f:
    model_hub_arch_class = json.load(f)

n2a_map = dict()
for arch_type, c in model_hub_arch_class.items():
    for lib, model_list in c.items():
        for model in model_list:
            n2a_map[model] = arch_type
#print(n2a_map)

dl = onnxzoo_classify(dl, n2a_map)
ll = onnxzoo_classify(ll, n2a_map)
pl = onnxzoo_classify(pl, n2a_map)


merge_vec(dl, ll, pl, pytorch_vision_path, 'torch/vision', n2a_map)
merge_vec(dl, ll, pl, keras_app_path, 'kerasapp', n2a_map)
merge_vec(dl, ll, pl, keras_nlp_path, 'kerasnlp', n2a_map)

path_hd = '././ptm_vectors/combined_vec/'

write_vec(dl, ll, pl, path_hd + 'vec_d.json', path_hd + 'vec_l.json', path_hd + 'vec_p.json')
'''

convert_to_pickle('././ptm_vectors/combined_vec/')