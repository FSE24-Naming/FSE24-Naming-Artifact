
import os
os.environ['KERAS_HOME'] = '<kerasapp_cache>' # change this to your own cache directory

import keras_nlp, json

import tensorflow as tf
import tf2onnx
import onnx
from collections import OrderedDict
from tf2onnx import optimizer


def keras_to_onnx(keras_model, directory):
    onnx_model, _ = tf2onnx.convert.from_keras(
        opset=16, 
        model=keras_model, 
        output_path=directory,
        optimizers = OrderedDict([
        ("optimize_transpose", optimizer.TransposeOptimizer),
        #("remove_redundant_upsample", optimizer.UpsampleOptimizer),
        ("fold_constants", optimizer.ConstFoldOptimizer),
        ("const_dequantize_optimizer", optimizer.ConstDequantizeOptimizer),
        ("loop_optimizer", optimizer.LoopOptimizer),
        # merge_duplication should be used after optimize_transpose
        # for optimize_transpose may have some trans nodes that can be merge
        ("merge_duplication", optimizer.MergeDuplicatedNodesOptimizer),
        #("reshape_optimizer", optimizer.ReshapeOptimizer),
        ("global_pool_optimizer", optimizer.GlobalPoolOptimizer),
        #("q_dq_optimizer", optimizer.QDQOptimizer),
        ("remove_identity", optimizer.IdentityOptimizer),
        #("remove_back_to_back", optimizer.BackToBackOptimizer),
        ("einsum_optimizer", optimizer.EinsumOptimizer),
        ])
    )
    return onnx_model

with open('./keras_nlp_backbone_variants.json') as f:
    bbv = json.load(f)

cur_m = os.listdir('./kerasnlp_onnx_f')

for arch_name, arch_var_list in bbv.items():
    model_class = getattr(keras_nlp.models, arch_name)
    for arch_var in arch_var_list:
        if arch_var + '.onnx' in cur_m:
            print(f'skipped {arch_var}')
            continue
        try:
            print(f'loading {arch_var}')
            model = model_class.from_preset(arch_var)
            print(f'converting {arch_var}')
            keras_to_onnx(model, '<kerasnlp_onnx_f/>' + arch_var + '.onnx') # change this to your own output directory
        except Exception as e:
            print(e)
