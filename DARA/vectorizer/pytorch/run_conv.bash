#!/bin/bash
module use /depot/davisjam/data/chingwo/general_env/modules
module load conda-env/general_env-py3.8.5
python ./vectorizer/pytorch/combine_all_hf_data.py