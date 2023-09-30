import os
import sys
import importlib.util
import inspect
import torch
import time

# onnx >= 1.14

TARGET_DIR = 'ANONYMIZED_PATH/pytorchhub_onnx_f'

'''
# specify the directory you want to search
root_dir = "ANONYMIZED_PATH/PTMTorrent/PTMTorrent/ptm_torrent/pytorchhub"

# recursively find all hubconf.py files
for foldername, subfolders, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename == 'hubconf.py':
            hubconf_path = os.path.join(foldername, filename)
            print(f"Found hubconf.py at {hubconf_path}")
'''
'''
import sys
sys.path.append('./onnxscript') # Replace with the actual path

import onnxscript

# Assuming you use opset17
from onnxscript.onnx_opset import opset18 as op

custom_opset = onnxscript.values.Opset(domain="torch.onnx", version=1)


@onnxscript.script(custom_opset)
def aten_unflatten(self, dim, sizes):
    """unflatten(Tensor(a) self, int dim, SymInt[] sizes) -> Tensor(a)"""

    self_size = op.Shape(self)

    if dim < 0:
        # PyTorch accepts negative dim as reversed counting
        self_rank = op.Size(self_size)
        dim = self_rank + dim

    head_start_idx = op.Constant(value_ints=[0])
    head_end_idx = op.Reshape(dim, op.Constant(value_ints=[1]))
    head_part_rank = op.Slice(self_size, head_start_idx, head_end_idx)

    tail_start_idx = op.Reshape(dim + 1, op.Constant(value_ints=[1]))
    #tail_end_idx = op.Constant(value_ints=[_INT64_MAX])
    tail_end_idx = op.Constant(value_ints=[9223372036854775807]) # = sys.maxint, exactly 2^63 - 1 -> 64 bit int
    tail_part_rank = op.Slice(self_size, tail_start_idx, tail_end_idx)

    final_shape = op.Concat(head_part_rank, sizes, tail_part_rank, axis=0)

    return op.Reshape(self, final_shape)

def custom_unflatten(g, *dim, **shape):
    return g.onnxscript_op(aten_unflatten, *dim, **shape)    

torch.onnx.register_custom_op_symbolic(
    symbolic_name="aten::unflatten",
    symbolic_fn=custom_unflatten,
    opset_version=18,
)
'''
IGNORE = {'get_model_weights', 'get_weight', 'mvit_v2_s', 'vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32'}
curr_repo = "pytorch/vision"
l = list(torch.hub.list(curr_repo))
print(len(l))
dummy_input = torch.randn(1, 3, 224, 224)
#image1, image2 = torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)
#dummy_input = (image1, image2)
d_f = os.listdir(TARGET_DIR)
for name in l:
    #print("converting", name)
    if name + ".onnx" in d_f or name in IGNORE:
        #print("skip", name)
        continue
    try:
        print("converting", name)
        model = torch.hub.load(curr_repo, name)
        model.eval()
        torch.onnx.export(model, dummy_input, TARGET_DIR + "/" + name + ".onnx", opset_version=18)
    except Exception as e:
        print(e)
        print(name, 'Failed')
        break
    #time.sleep(15)
        