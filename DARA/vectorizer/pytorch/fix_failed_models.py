from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor, AutoProcessor
import torch, json, os, traceback, sys
from list_gen import OrderedListGenerator
from list_to_json import node_list_to_json
from datasets import load_dataset
import numpy as np

with open('././ptm_vectors/failed_models_reason.json') as f:
    fm_data = json.load(f)
with open('./model_collection/filtered_models.json') as f:
    data = json.load(f)
all_m = [model_name for arch_list in data.values() for model_name in arch_list]


d = "ANONYMIZED_PATH/cache_huggingface"
sys.setrecursionlimit(5000)
def specify_enc_dec_fix():

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

    total_len = len(fm_data.keys())
    curr_pos = 0
    for n in fm_data.keys():
        with open('././failed_model_fixing_file/specify_enc_dec_fix.json') as f:
            cfm = json.load(f)
        print('[{}/{}] Fixing'.format(curr_pos, total_len), n)
        curr_pos += 1
        modified_model_name = ''
        for ch in n:
            if ch == '/':
                nch = '>'
            else:
                nch = ch
            modified_model_name += nch
        if os.path.exists('././ptm_data' + '/' + modified_model_name + '.json'):
            print('././ptm_data' + '/' + modified_model_name + '.json', 'exists, skipped.')
            continue
        if fm_data[n][0] == 'Model Loading':
            print('Unrelavent error for', n, ', Skipped')
            continue
        if n in MANUAL_SKIP or n in cfm:
            print('Manually skipped', n)
            continue
        try:
            m = AutoModel.from_pretrained(n, cache_dir = d)
            print('Model Generated')
        except Exception as e:
            print('Failed to generate model')
            if 'CUDA' in str(e): break
            with open('././failed_model_fixing_file/specify_enc_dec_fix.json', 'w') as f:
                json.dump(cfm + [n], f)
            continue
        try:
            t = AutoTokenizer.from_pretrained(n, cache_dir = d)
            print('Tokenizer Generated')
        except Exception as e:
            print('Failed to generate tokenizer')
            if 'CUDA' in str(e): break
            with open('././failed_model_fixing_file/specify_enc_dec_fix.json', 'w') as f:
                json.dump(cfm + [n], f)
            continue
        try:
            input_ids = t('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            attention_mask = input_ids.ne(m.config.pad_token_id).long()
            decoder_input_ids = t('<pad> <extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
            print('Input generated')
        except Exception as e:
            print('Failed to generate inputs')
            traceback.print_exc()
            if 'CUDA' in str(e): break
            with open('././failed_model_fixing_file/specify_enc_dec_fix.json', 'w') as f:
                json.dump(cfm + [n], f)
            continue
        
        try:
            gen = OrderedListGenerator(m, (input_ids, attention_mask, decoder_input_ids), use_hash=True)
            l, c = gen.get_connection()
            node_list_to_json(l, c, '././ptm_data' + '/' + modified_model_name + '.json')
        except:
            print('Failed to generate ordered list')
            traceback.print_exc()
            with open('././failed_model_fixing_file/specify_enc_dec_fix.json', 'w') as f:
                json.dump(cfm + [n], f)

def from_tf_fix():

    MANUAL_SKIP = {
        "mrm8488/convbert-small-spanish",
        "mrm8488/convbert-base-spanish",
        "inverse-scaling/opt-30b_eval",
        "google/switch-xxl-128"
    }

    total_len = len(fm_data.keys())
    curr_pos = 0
    for n in fm_data.keys():
        with open('././failed_model_fixing_file/from_tf_fix.json') as f:
            cfm = json.load(f)
        print('[{}/{}] Fixing'.format(curr_pos, total_len), n)
        curr_pos += 1
        modified_model_name = ''
        for ch in n:
            if ch == '/':
                nch = '>'
            else:
                nch = ch
            modified_model_name += nch
        if os.path.exists('././ptm_data' + '/' + modified_model_name + '.json'):
            print('././ptm_data' + '/' + modified_model_name + '.json', 'exists, skipped.')
            continue
        if fm_data[n][0] != 'Model Loading':
            print('Unrelavent error for', n, ', Skipped')
            continue
        if n in MANUAL_SKIP or n in cfm:
            print('Manually skipped', n)
            continue
        try:
            m = AutoModel.from_pretrained(n, cache_dir = d)
            print('Model Generated')
        except Exception as e:
            print('Failed to generate model')
            traceback.print_exc()
            if 'CUDA' in str(e): break
            with open('././failed_model_fixing_file/from_tf_fix.json', 'w') as f:
                json.dump(cfm + [n], f)
            continue
        try:
            t = AutoTokenizer.from_pretrained(n, cache_dir = d, from_tf=True)
            print('Tokenizer Generated')
        except Exception as e:
            print('Failed to generate tokenizer')
            traceback.print_exc()
            if 'CUDA' in str(e): break
            with open('././failed_model_fixing_file/from_tf_fix.json', 'w') as f:
                json.dump(cfm + [n], f)
            continue
        try:
            inp = t('Test', return_tensors='pt')
            print('Input generated')
        except Exception as e:
            print('Failed to generate inputs')
            traceback.print_exc()
            if 'CUDA' in str(e): break
            with open('././failed_model_fixing_file/from_tf_fix.json', 'w') as f:
                json.dump(cfm + [n], f)
            continue
        
        try:
            gen = OrderedListGenerator(m, inp, use_hash=True)
            l, c = gen.get_connection()
            node_list_to_json(l, c, '././ptm_data' + '/' + modified_model_name + '.json')
        except:
            print('Failed to generate ordered list')
            traceback.print_exc()
            with open('././failed_model_fixing_file/from_tf_fix.json', 'w') as f:
                json.dump(cfm + [n], f)

def missing_raw_speech_fix():
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

    total_len = len(fm_data.keys())
    curr_pos = 0
    #for n in fm_data.keys():
    for n in all_m:
        with open('././failed_model_fixing_file/missing_raw_speech_fix.json') as f:
            cfm = json.load(f)
        print('[{}/{}] Fixing'.format(curr_pos, total_len), n)
        curr_pos += 1
        modified_model_name = ''
        for ch in n:
            if ch == '/':
                nch = '>'
            else:
                nch = ch
            modified_model_name += nch
        if os.path.exists('././ptm_data' + '/' + modified_model_name + '.json'):
            print('././ptm_data' + '/' + modified_model_name + '.json', 'exists, skipped.')
            continue
        '''
        if fm_data[n][0] == 'Model Loading':
            print('Unrelavent error for', n, ', Skipped')
            continue'''
        if n in MANUAL_SKIP:
            print('Manually skipped', n)
            continue
        if n in cfm:
            print('Existed, skipped', n)
            continue
        try:
            m = AutoModel.from_pretrained(n, cache_dir = d)
            print('Model Generated')
        except Exception as e:
            print('Failed to generate model')
            traceback.print_exc()
            if 'CUDA' in str(e): break
            with open('././failed_model_fixing_file/missing_raw_speech_fix.json', 'w') as f:
                json.dump(cfm + [n], f)
            continue
        try:
            t = AutoFeatureExtractor.from_pretrained(n, cache_dir = d)
            print('FeatureExtractor Generated')
        except Exception as e:
            print('Failed to generate FeatureExtractor')
            traceback.print_exc()
            if 'CUDA' in str(e): break
            with open('././failed_model_fixing_file/missing_raw_speech_fix.json', 'w') as f:
                json.dump(cfm + [n], f)
            continue
        try:
            inp = t(np.random.randn(1, 16000), sampling_rate=16000, return_tensors='pt')
            print('Input generated')
        except Exception as e:
            print('Failed to generate inputs')
            traceback.print_exc()
            if 'CUDA' in str(e): break
            with open('././failed_model_fixing_file/missing_raw_speech_fix.json', 'w') as f:
                json.dump(cfm + [n], f)
            continue
        
        try:
            gen = OrderedListGenerator(m, inp, use_hash=True)
            l, c = gen.get_connection()
            node_list_to_json(l, c, '././ptm_data' + '/' + modified_model_name + '.json')
        except:
            print('Failed to generate ordered list')
            traceback.print_exc()
            with open('././failed_model_fixing_file/missing_raw_speech_fix.json', 'w') as f:
                json.dump(cfm + [n], f)

def type_diff_fix():

    MANUAL_SKIP = {
        "mrm8488/convbert-small-spanish",
        "mrm8488/convbert-base-spanish",
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

    total_len = len(fm_data.keys())
    curr_pos = 0
    for n in fm_data.keys():
        print('[{}/{}] Fixing'.format(curr_pos, total_len), n)
        curr_pos += 1
        modified_model_name = ''
        for ch in n:
            if ch == '/':
                nch = '>'
            else:
                nch = ch
            modified_model_name += nch
        if os.path.exists('././ptm_data' + '/' + modified_model_name + '.json'):
            print('././ptm_data' + '/' + modified_model_name + '.json', 'exists, skipped.')
            continue
        if fm_data[n][0] == 'Model Loading':
            print('Unrelavent error for', n, ', Skipped')
            continue
        if n in MANUAL_SKIP:
            print('Manually skipped', n)
            continue
        try:
            m = AutoModel.from_pretrained(n, cache_dir = d)
            print('Model Generated')
        except Exception as e:
            print('Failed to generate model')
            if 'CUDA' in str(e): break
            continue
        try:
            t = AutoTokenizer.from_pretrained(n, cache_dir = d)
            print('Tokenizer Generated')
        except Exception as e:
            print('Failed to generate tokenizer')
            if 'CUDA' in str(e): break
            continue
        try:
            input_ids = t('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            attention_mask = input_ids.ne(m.config.pad_token_id).long()
            decoder_input_ids = t('<pad> <extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
            print('Input generated')
        except Exception as e:
            print('Failed to generate inputs')
            traceback.print_exc()
            if 'CUDA' in str(e): break
            continue
        
        try:
            gen = OrderedListGenerator(m, (input_ids, attention_mask, decoder_input_ids))
            l, c = gen.get_connection()
            node_list_to_json(l, c, '././ptm_data' + '/' + modified_model_name + '.json')
        except:
            print('Failed to generate ordered list')
            traceback.print_exc()

import pandas as pd

def pd_input_fix():

    MANUAL_SKIP = {
        "mrm8488/convbert-small-spanish",
        "mrm8488/convbert-base-spanish",
        "inverse-scaling/opt-30b_eval",
        "google/switch-xxl-128"
    }

    total_len = len(fm_data.keys())
    curr_pos = 0
    for n in fm_data.keys():
        with open('././failed_model_fixing_file/pd_input_fix.json') as f:
            cfm = json.load(f)
        print('[{}/{}] Fixing'.format(curr_pos, total_len), n)
        curr_pos += 1
        modified_model_name = ''
        for ch in n:
            if ch == '/':
                nch = '>'
            else:
                nch = ch
            modified_model_name += nch
        if os.path.exists('././ptm_data' + '/' + modified_model_name + '.json'):
            print('././ptm_data' + '/' + modified_model_name + '.json', 'exists, skipped.')
            continue
        if fm_data[n][0] != 'Model Loading':
            print('Unrelavent error for', n, ', Skipped')
            continue
        if n in MANUAL_SKIP or n in cfm:
            print('Manually skipped', n)
            continue
        try:
            m = AutoModel.from_pretrained(n, cache_dir = d)
            print('Model Generated')
        except Exception as e:
            print('Failed to generate model')
            traceback.print_exc()
            if 'CUDA' in str(e): break
            with open('././failed_model_fixing_file/pd_input_fix.json', 'w') as f:
                json.dump(cfm + [n], f)
            continue
        try:
            t = AutoProcessor.from_pretrained(n, cache_dir = d, from_tf=True)
            print('Tokenizer Generated')
        except Exception as e:
            print('Failed to generate tokenizer')
            traceback.print_exc()
            if 'CUDA' in str(e): break
            with open('././failed_model_fixing_file/pd_input_fix.json', 'w') as f:
                json.dump(cfm + [n], f)
            continue
        try:
            inp = t(pd.DataFrame(), return_tensors='pt')
            print('Input generated')
        except Exception as e:
            print('Failed to generate inputs')
            traceback.print_exc()
            if 'CUDA' in str(e): break
            with open('././failed_model_fixing_file/pd_input_fix.json', 'w') as f:
                json.dump(cfm + [n], f)
            continue
        
        try:
            gen = OrderedListGenerator(m, inp, use_hash=True)
            l, c = gen.get_connection()
            node_list_to_json(l, c, '././ptm_data' + '/' + modified_model_name + '.json')
        except:
            print('Failed to generate ordered list')
            traceback.print_exc()
            with open('././failed_model_fixing_file/pd_input_fix.json', 'w') as f:
                json.dump(cfm + [n], f)

def audio_0_fix():

    MANUAL_SKIP = {
        "mrm8488/convbert-small-spanish",
        "mrm8488/convbert-base-spanish",
        "inverse-scaling/opt-30b_eval",
        "google/switch-xxl-128"
    }

    total_len = len(fm_data.keys())
    curr_pos = 0
    for n in fm_data.keys():
        with open('././failed_model_fixing_file/audio_0_fix.json') as f:
            cfm = json.load(f)
        print('[{}/{}] Fixing'.format(curr_pos, total_len), n)
        curr_pos += 1
        modified_model_name = ''
        for ch in n:
            if ch == '/':
                nch = '>'
            else:
                nch = ch
            modified_model_name += nch
        if os.path.exists('././ptm_data' + '/' + modified_model_name + '.json'):
            print('././ptm_data' + '/' + modified_model_name + '.json', 'exists, skipped.')
            continue
        if fm_data[n][0] != 'Model Loading':
            print('Unrelavent error for', n, ', Skipped')
            continue
        if n in MANUAL_SKIP or n in cfm:
            print('Manually skipped', n)
            continue
        try:
            m = AutoModel.from_pretrained(n, cache_dir = d)
            print('Model Generated')
        except Exception as e:
            print('Failed to generate model')
            traceback.print_exc()
            if 'CUDA' in str(e): break
            with open('././failed_model_fixing_file/audio_0_fix.json', 'w') as f:
                json.dump(cfm + [n], f)
            continue
        try:
            t = AutoProcessor.from_pretrained(n, cache_dir = d)
            print('Processor Generated')
        except Exception as e:
            print('Failed to generate processor')
            traceback.print_exc()
            if 'CUDA' in str(e): break
            with open('././failed_model_fixing_file/audio_0_fix.json', 'w') as f:
                json.dump(cfm + [n], f)
            continue
        try:
            inp = t(np.random.randn(1, 16000), sampling_rate=16000, return_tensors='pt')
            print('Input generated')
        except Exception as e:
            print('Failed to generate inputs')
            traceback.print_exc()
            if 'CUDA' in str(e): break
            with open('././failed_model_fixing_file/audio_0_fix.json', 'w') as f:
                json.dump(cfm + [n], f)
            continue
        
        try:
            gen = OrderedListGenerator(m, inp, use_hash=True)
            l, c = gen.get_connection()
            node_list_to_json(l, c, '././ptm_data' + '/' + modified_model_name + '.json')
        except:
            print('Failed to generate ordered list')
            traceback.print_exc()
            with open('././failed_model_fixing_file/audio_0_fix.json', 'w') as f:
                json.dump(cfm + [n], f)

specify_enc_dec_fix()
#missing_raw_speech_fix()
#missing_raw_speech_fix()