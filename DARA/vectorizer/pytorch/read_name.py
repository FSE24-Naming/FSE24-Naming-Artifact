import json, os

'''
with open('././ptm_vectors/onnx_vec/vec_d.json') as f:
    data = json.load(f)

d = sorted(list(data['OnnxModelZoo'].keys()))

for i in d: 
    print('\"'+i+'\",')'''

l = sorted(list(os.listdir('ANONYMIZED_PATH/kerasapp_onnx_f')))
for i in l:
    if i[-5:] == '.onnx':
        print('\"'+i[:-5]+'\",')