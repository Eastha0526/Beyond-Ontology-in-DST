# ATOM

Here's the link to the archive:

> [Advancing Tasked-Oriented Dialogue with GNN and LLM: From A Link Prediction Perspective]()

## Overview

Traditional Dialog State Tracking (DST) systems struggle with multi-domain and complex conversations. To address this, we propose the ATOM model, which combines large language models (LLMs) and graph neural networks (GNNs) for task-oriented dialogue. Our approach uses instruction tuning and zero-shot inference for semantic DST, effectively capturing user intents. GNNs graphically represent the dialogue state and predict slot-value links, improving slot determination with minimal user interaction. Experimental results show that ATOM excels in distinguishing various intents and performs well with heuristic and latent features, making it suitable for real-world applications and enhancing slot value organization in user utterances.

## Environment Setup

```
conda create -n ATOM python=3.10
conda activate ATOM
pip install -r requirements.txt
```

## Data preparation

The three benchmark Datasets can be downloaded at:

- MultiWOZ 2.0 : https://github.com/budzianowski/multiwoz
- MultiWOZ 2.4 : https://github.com/smartyfh/MultiWOZ2.4
- SGD : https://github.com/google-research-datasets/dstc8-schema-guided-dialogue

We were provided with preprocessing scripts and base scripts by [LDST](https://github.com/WoodScene/LDST).

## DST Training (`finetune.py`)

```
python3 finetune.py --base_model 'meta-llama/Meta-Llama-3-8B' --data_path './Data/MULTIWOZ2.4_preprocess/train_LLM_zero-shot.json' --output_dir './ATOM_Results_zero-shot_MultiWOZ2.4' --num_epochs=2 --micro_batch_size=8
```
Training on a single Nvidia 4090 GPU is expected to take approximately 180 hours. Upon completion, the fine-tuned model weights will be saved in `$output_dir`.

## DST Inference (`generate_zero_shot.py`)

You can load the provided weights directly from the \checkpoint folder and perform inference.

How to make inference:

```
python3 generate_zero_shot.py --load_8bit = True --base_model = 'meta-llama/Meta-Llama-3-8B' --lora_weights './ATOM_Results_zero-shot_MultiWOZ2.4' --testfile_name './Data/MULTIWOZ2.4_preprocess/test.json' --testfile_idx './Data/MULTIWOZ2.4_preprocess/test.idx' --output_file './ATOM_result/MULTIWOZ2.4.txt'
```

## DST Evaluation

```
python3 evalutation.py --data_dir '$DATA_DIR' --output_dir '$DATA_DIR/output/' --test_idx '$DATA_DIR/test.idx'
```

## GNN Training

To train the gnn with the VGAE model we created 

```
python3 train_gnn.py --data_dir '$DATA_DIR' --output_dir '$OUTPUT_DIR'
```

## ATOM

To run an atom model that predicts intent within the following conversation 

```
python3 ATOM.py
```

## Acknowledgements

1. [LDST]() : upon which our overall code is built.
2. [FastChat]() : we borowwed the chat templates from this repository.

They have been a great addition to our research.

## Citation

If you can cite [our paper]():

```
@article{lee2024atom,
title={Advancing Tasked-Oriented Dialogue with GNN and LLM: From A Link Prediction Perspective},
author={Sejin Lee, Dongha Kim, and Min Song},
journal={arXiv preprint },
year={2024}
}
```
