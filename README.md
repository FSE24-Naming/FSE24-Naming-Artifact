# ESEC/FSE 2024 Artifact Repository

> Artifact repository for paper "Naming Conventions and Defects in Pre-trained Models"

## About

This repository contains all of the elements of the paper.

### Repo-view

The next table describes the "table of contents" for this repository.

| Top-level folder |  Second-level folder |
|------------------|-----------------------------|
| `PTM Collection/`  | 'hf_data_analyzer/'             |
|                    |  'model_collection/'     |
|                    |  'converters/'      |
| `DARA/`            | 'vectorizer/'             |
|                    |  'model_clustering'       |
| `name_analysis/`   | 'manual_labeling/'        |


## [PTM Collection](\PTM_collection) ($5)

- [hf_data_analyzer](\PTM_collection\hf_data_analyzer): 
    - The scripts to anlayzed the PTM data from HuggingFace repository.

- [model_collection](\PTM_collection\model_collection) ($5.2.1):
    - The scripts used to collect the PTM names, downloads, and the metadata from HuggingFace repository.

- [converters](\PTM_collection\converters) ($5.2.2): 
   -  The scripts used to convert the gated models from Keras, TensorFlow, and PyTorch to the ONNX format.
   - The tests used to verify the converters.



## [DNN ARchitecture Assessment (DARA)](\DARA) ($7)
- [vectorizer](\DARA\vectorizer)($7.2):
    - This folder contains the scripts used to vectorize the models.

- [model_clustering](\DARA\model_clustering)  ($7.3):
    - This folder contains the scripts used to cluster the models.

## [Name Analysis](\name_analysis)  ($6, $8)
- [pattern_analysis](\name_analysis\pattern_analysis) ($6):
    -  This folder contains the scripts used to analyze the semantic ($6.2.1) and syntactic ($6.2.2) naming patterns.

- [defect_analysis](\name_analysis\defect_analysis) ($8):
    - This folder contains the scripts used to analyze the defects.

- [manual_labeling](\name_analysis\manual_labeling) ($8):
    - Evaluation of GPT-4 output ($6.1)
    - Manual labeling of outliers ($8.2)

