from data_cleaner import read_from_json
import pandas as pd
import random

def dict_simplify(model_name_dict):
    simp_dict = dict()
    for model_name, seg_cat_conf_dict in model_name_dict.items():
        for seg, cat_conf_pairs in seg_cat_conf_dict.items():
            if cat_conf_pairs == None: continue
            top1 = cat_conf_pairs[0][0]
            if model_name not in simp_dict:
                simp_dict[model_name] = []
            simp_dict[model_name].append((seg, top1))
    return simp_dict

def dict_filter(s_model_name_dict, cat):
    fil_s_model_name_dict = dict()
    for model_name, seg_top1s in s_model_name_dict.items():
        for seg_top1 in seg_top1s:
            if seg_top1[1] == cat:
                if model_name not in fil_s_model_name_dict:
                    fil_s_model_name_dict[model_name] = []
                fil_s_model_name_dict[model_name] = seg_top1s
    return fil_s_model_name_dict

def random_selector(fil_s_model_name_dict, count):
    if count >= len(fil_s_model_name_dict.keys()):
        count = len(fil_s_model_name_dict.keys())
    selected = random.sample(list(fil_s_model_name_dict.items()), count)
    return dict(selected)

def get_df(rand_s_model_name_dict, target, arch_map):
    columns = ['Model Name', 'Pattern', 'Target', 'Arch']
    df = pd.DataFrame(columns=columns)
    cnt = 0
    for k, v in rand_s_model_name_dict.items():
        df.loc[len(df)] = ({'Model Name': k, 'Pattern': str(v), 'Target': target, 'Arch': arch_map[k]})
        cnt += 1
    return df

mnd = read_from_json('final_cleaned_output.json')
n_a_map = read_from_json('name_arch_map.json')
smnd = dict_simplify(mnd)
fsmnd_A = dict_filter(smnd, 'A')
fsmnd_S = dict_filter(smnd, 'S')
fsmnd_V = dict_filter(smnd, 'V')
fsmnd_D = dict_filter(smnd, 'D')
fsmnd_T = dict_filter(smnd, 'T')
fsmnd_F = dict_filter(smnd, 'F')
fsmnd_L = dict_filter(smnd, 'L')
fsmnd_O = dict_filter(smnd, 'O')

COUNT = 50
rand_A = random_selector(fsmnd_A, COUNT)
rand_S = random_selector(fsmnd_S, COUNT)
rand_V = random_selector(fsmnd_V, COUNT)
rand_D = random_selector(fsmnd_D, COUNT)
rand_T = random_selector(fsmnd_T, COUNT)
rand_F = random_selector(fsmnd_F, COUNT)
rand_L = random_selector(fsmnd_L, COUNT)
rand_O = random_selector(fsmnd_O, COUNT)

df_A = get_df(rand_A, 'A', n_a_map)
df_S = get_df(rand_S, 'S', n_a_map)
df_V = get_df(rand_V, 'V', n_a_map)
df_D = get_df(rand_D, 'D', n_a_map)
df_T = get_df(rand_T, 'T', n_a_map)
df_F = get_df(rand_F, 'F', n_a_map)
df_L = get_df(rand_L, 'L', n_a_map)
df_O = get_df(rand_O, 'O', n_a_map)

mnd_wa_raw = read_from_json('wrong_arch_wa.json')

mnd_wa = dict()
for mn in mnd_wa_raw:
    mnd_wa[mn] = mnd[mn]
rand_Af = random_selector(mnd_wa, COUNT)
df_Af = get_df(rand_Af, 'Af', n_a_map)

frames = [df_A, df_S, df_V, df_D, df_T, df_F, df_L, df_O]
result = pd.concat(frames)
result_Af = df_Af

result.to_csv('random_selected_models.csv')
result_Af.to_csv('random_selected_Af_models.csv')

