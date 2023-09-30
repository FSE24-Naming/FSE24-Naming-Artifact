import json
import traceback
import traceback
from list_gen import OrderedListGenerator
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor, AutoProcessor
from PIL import Image
import numpy as np

with open('././ptm_vectors/failed_models_subset_2.json') as f:
    data = json.load(f)

#model_names = list(data.keys())
model_names = data

DIR = '././ptm_vectors/failed_models_reason_subset_2.json'
CACHE_DIR = "ANONYMIZED_PATH/cache_huggingface"
IMAGE = Image.open('././000000039769.jpg')

def test_tracing(m, i):
    try:
        gen = OrderedListGenerator(m, i, use_hash=True)
        gen.get_connection()
    except Exception:
        return traceback.format_exc()

def test_loading(n):
    try:
        m = AutoModel.from_pretrained(n, cache_dir = CACHE_DIR)
    except Exception:
        return traceback.format_exc()
    t = []
    try:
        p = AutoProcessor.from_pretrained(n, cache_dir = CACHE_DIR)
        print('Process Successsful')
        i = p("Test Input", return_tensors="pt")
        return m, i
    except Exception as e:
        t.append(traceback.format_exc())
    try:
        p = AutoTokenizer.from_pretrained(n, cache_dir = CACHE_DIR)
        print('Tokenizing Successsful')
        i = p.encode("Test Input", return_tensors="pt")
        return m, i
    except Exception as e:
        t.append(traceback.format_exc())
    try:
        p = AutoImageProcessor.from_pretrained(n, cache_dir = CACHE_DIR)
        print('ImageProcessor Successsful')
        i = p(images=IMAGE, return_tensors="pt")
        return m, i
    except Exception as e:
        t.append(traceback.format_exc())
    try:
        p = AutoFeatureExtractor.from_pretrained(n, cache_dir = CACHE_DIR)
        print('FeatureExtractor Successsful')
        i = p(np.random.randn(1, 16000), sampling_rate=16000, return_tensors='pt')
        return m, i
    except Exception as e:
        t.append(traceback.format_exc())
    return t

d = dict()
for i in range(50):
    print('[{}] TESTING'.format(i), model_names[i])
    obj = test_loading(model_names[i])
    if type(obj) == tuple:
        m, ip = obj
        msg = test_tracing(m, ip)
        tbl = msg.split('\n')
        el = []
        for m in tbl:
            if 'Error: ' in m and m != 'RuntimeError: Failed to run torchgraph see error message':
                el.append(m)
        d[model_names[i]] = el
    elif type(obj) == list:
        ell = []
        for msg in obj:
            tbl = msg.split('\n')
            el = []
            for m in tbl:
                if 'Error: ' in m:
                    el.append(m)
            ell.append(el)
        eld = {'AutoProcessor': ell[0], 'AutoTokenizer': ell[1], 'AutoImageProcessor': ell[2], 'AutoFeatureExtractor': ell[3]}
        d[model_names[i]] = eld
    else:
        msg = obj
        tbl = msg.split('\n')
        el = []
        for m in tbl:
            if 'Error: ' in m:
                el.append(m)
        d[model_names[i]] = el

    with open(DIR, 'w') as f:
        json.dump(d, f)
        