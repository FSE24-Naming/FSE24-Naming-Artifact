# For keras application

import tensorflow as tf
import tf2onnx
import onnx
from collections import OrderedDict
from tf2onnx import optimizer

import os

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

import tensorflow.keras.applications as apps
import os, pkgutil, inspect 

# This gives the path of the installed module
path = apps.__path__[0]

# This will give you a list of all modules in the given path
sub_modules = [name for _, name, _ in pkgutil.iter_modules([path])]

for sub_module_name in sub_modules:

    # Dynamically import the sub-module using its name
    sub_module = __import__(f"tensorflow.keras.applications.{sub_module_name}", fromlist=[sub_module_name])

    smn_wo_us = sub_module_name.split("_")
    
    functions = []

    # Get all functions in the sub-module using inspect
    for name, obj in inspect.getmembers(sub_module):
        if inspect.isfunction(obj):
            name_lc = name.lower()
            for item in smn_wo_us:
                if item in name_lc:
                    functions.append(obj)
                    break