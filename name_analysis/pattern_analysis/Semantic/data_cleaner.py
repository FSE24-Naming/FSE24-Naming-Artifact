
from loguru import logger
import json

CATEGORIES = ['A', 'S', 'D', 'C', 'V', 'F', 'L', 'T', 'R', 'N', 'H', 'P', 'O']

def parse_text(file_name):

    with open(file_name) as f:
        content = f.read()

    sep_content = content.split('\n')
    model_name_dict = dict()
    repeated_model_name_set = set()

    for content_segment in sep_content: 

        cat_conf_dict = dict()

        if content_segment == (''): break

        curr_idx = content_segment.find(' ')
        model_name = content_segment[:curr_idx]
        first = True

        while True:
            if first:
                curr_idx = content_segment.find(' ') - 1
                first = False
            else:
                curr_idx = content_segment.find('),')
            if curr_idx == -1: break
            content_segment = content_segment[curr_idx+2:]

            curr_idx = content_segment.find(':')
            name_comp = content_segment[:curr_idx]
            try:
                content_segment = content_segment[curr_idx+1:]
            except:
                logger.error(f'In {model_name}: Cannot parse string due to a content segment do not contain categories and confidences.')
                return None
            if curr_idx == -1: break
            
            cat_conf = parse_category_confidence(content_segment, model_name)
            cat_conf_dict[name_comp] = (cat_conf)

        if model_name in model_name_dict:
            logger.warning(f'Repeated model name {model_name}. Previous category confidence {model_name_dict[model_name]} discarded.')
            repeated_model_name_set.add(model_name)
        
        model_name_dict[model_name] = cat_conf_dict

    return model_name_dict, repeated_model_name_set


def parse_category_confidence(content_segment, model_name):
    
    top_1, top_2, top_3 = None, None, None
    try:
        top_1 = (content_segment[1], float(content_segment[3:6]))
    except:
        logger.error(f'In {model_name}: Cannot parse string due to a content segment do not contain categories and confidences.')
        return None
    try:
        top_2 = (content_segment[8], float(content_segment[10:13]))
    except:
        logger.warning(f'In {model_name}: Did not find top_2 category.')
        return [top_1, top_2, top_3]
    try:
        top_3 = (content_segment[15], float(content_segment[17:20]))
    except:
        logger.warning(f'In {model_name}: Did not find top_3 category.')
        return [top_1, top_2, top_3]
    try:
        assert top_1[0] in CATEGORIES
        return [top_1, top_2, top_3]
        # assert top_2[0] in CATEGORIES
        # assert top_3[0] in CATEGORIES
    except:
        logger.error(f'In {model_name}: Invalid category.')
        return None
    
def write_to_json(d, dir):
    with open(dir, 'w') as f:
        json.dump(d, f)
def write_set_to_json(s, dir):
    with open(dir, 'w') as f:
        json.dump(list(s), f)
def read_from_json(dir):
    with open(dir) as f:
        d = json.load(f)
    return d

def find_models_to_run(model_name_dict):
    with open('name_order.json') as f:
        name_order = json.load(f)
    models_to_run = []
    for name in name_order:
        if name not in model_name_dict:
            models_to_run.append(name)
    return models_to_run

def find_mismatch(model_name_dict):
    err_cnt = 0
    mismatch_model_list = []
    for model_name, model_comp in model_name_dict.items():
        orig_model_name = model_name
        model_name = model_name.split('/')[-1]
        for model_comp_str, cat_conf_list in model_comp.items():
            model_name = model_name.replace(model_comp_str, '')
        model_name = model_name.replace('_', '')
        model_name = model_name.replace('-', '')
        if model_name != '':
            logger.error(f'Wrong model name segment in {orig_model_name}')
            err_cnt += 1
            mismatch_model_list.append(orig_model_name)
    logger.success(f'Find total mismatches: {err_cnt}')
    return mismatch_model_list

def combine_json(dirold, dirnew, outdir):
    with open(dirold) as f:
        dold = json.load(f)
    with open(dirnew) as f:
        dnew = json.load(f)
    for k, v in dnew.items():
        dold[k] = v
    with open(outdir, 'w') as f:
        json.dump(dold, f)

def find_name_repeat_in_seg(d):
    ret = []
    for k, v in d.items():
        if len(v.keys()) == 1:
            if list(v.keys())[0] == k.split('/')[-1]:
                logger.warning('Find model segment with repeated name')
                logger.info(f'Model: {k}')
                ret.append(k)
    return ret

def remove_false_repeat(d):
    removed = []
    for k, v in d.items():
        for kk, vv in v.items():
            #upper_cnt = sum(1 for char in kk if char.isupper())
            slash_cnt = sum(1 for char in kk if char == '-')
            udsc_cnt = sum(1 for char in kk if char == '_')
            if not(slash_cnt > 0 or udsc_cnt > 0):
                removed.append(k)
    return removed


if __name__ == '__main__':
    #model_name_dict, repeated_models = parse_text('response_new.txt')
    #mml = find_mismatch(model_name_dict)
    #write_to_json(mml, 'mismatched_models.json')
    #write_to_json(model_name_dict, 'output_name.json')
    #write_set_to_json(repeated_models, 'output_repeated_models.json')
    #models_to_run = find_models_to_run(model_name_dict)
    #write_to_json(models_to_run, 'output_models_to_run.json')
    # combine_json('output_name_old0.json', 'output_name.json', 'combined_output.json')
    #temp, _ = parse_text('response_ADTO.txt')
    #write_to_json(temp, 'output_ADTO.json')

    #combine_json('combined_output.json', 'output_ADTO.json', 'combined_output_new.json')
    #d = read_from_json('combined_output.json')
    #print(len(d.keys()))
    #model_name_dict, _ = parse_text('response_new1.txt')
    #write_to_json(model_name_dict, 'temp.json')
    #d, _ = parse_text('response_new0.txt')
    #find_mismatch(d)
    #write_to_json(d, 'temp.json')
    #combine_json('temp.json', 'output_name.json', 'combined_output.json')
    #co = read_from_json('combined_output.json')
    #m = find_mismatch(co)
    #cco = dict()
    #for k, v in co.items():
    #    if k not in m:
    #        cco[k] = v
    #write_to_json(cco, 'cleaned_combined_output.json')
    # em, _ = parse_text('gpt4_output_external_model.txt')
    # write_to_json(em, 'external_model_name.json')
    # pass
    #d = read_from_json('combined_output.json')
    #l = find_name_repeat_in_seg(d)
    #d2 = dict()
    #for ls in l:
    #    d2[ls] = d[ls]
    #for k, v in d2.items():
    #    print(k, v)
    #lll = remove_false_repeat(d2)
    #write_to_json(lll, 'FP_repeated_name_in_seg.json')
    #write_to_json(l, 'repeated_name_in_seg.json')
    '''
    mnd = read_from_json('combined_output.json')
    mnd_wa_raw = read_from_json('wrong_arch.json')
    mnd_wa = dict()
    for mn in mnd_wa_raw:
        mnd_wa[mn] = mnd[mn]
    FP_rnis = read_from_json('FP_repeated_name_in_seg.json')
    rnis = read_from_json('repeated_name_in_seg.json')
    fco = dict()
    for k, v in mnd_wa.items():
        if not(k in rnis and k not in FP_rnis):
            fco[k] = v
    write_to_json(fco, 'wrong_arch_wa.json')'''
    d = read_from_json('wrong_arch_wa.json')
    print(len(d))
