# Beyond Ontology in Dialogue State Tracking for Goal-Oriented Chatbot

## Overview

![figure2(2)](https://github.com/Eastha0526/ATOM/assets/110336043/e7758bc3-8614-4a7e-8e0a-8c4c987a128d)
Goal-oriented chatbots play a key role in automating user tasks such as booking flights or making restaurant reservations. The key to the effectiveness of these systems is Dialogue State Tracking (DST), which captures user intent and the state of the conversation. Existing DST approaches that rely on fixed ontologies and manual slot value compilation have suffered from a lack of adaptability and open domain applicability. To address this, we propose a novel approach to enhance DST using instruction tuning and prompting strategy without ontology. In the prompt strategy stage, we design optimal DST prompts to enable LLM to make thought-based inferences and incorporate an anti-hallucination stage to accurately track dialogue state and user intent in diverse conversation. Furthermore, the proposed Variational Graph Auto-Encoder (VGAE) stage ensures DST accuracy based on dialogue context and intentions. The approach achieved state-of-the-art with a JGA of 42.57\% compared to the ontology-less DST model, which even outperformed on open-domain real-world conversations. This work represents a significant advance in DST, enabling more adaptive and accurate goal-oriented chatbots.
## Environment Setup

```
conda create -n BODST python=3.10
conda activate BODST
pip install -r requirements.txt
```

## Data preparation

The three benchmark Datasets can be downloaded at:

- MultiWOZ 2.0 : https://github.com/budzianowski/multiwoz
- MultiWOZ 2.4 : https://github.com/smartyfh/MultiWOZ2.4
- SGD : https://github.com/google-research-datasets/dstc8-schema-guided-dialogue
- Persona Chat : https://www.kaggle.com/datasets/atharvjairath/personachat

We were provided with preprocessing scripts and base scripts by [LDST](https://github.com/WoodScene/LDST).

## DST Training (`finetune.py`)

```ruby
python3 finetune.py --base_model 'meta-llama/Meta-Llama-3-8B' \
                    --data_path '$DATA_DIR' \
                    --output_dir '$OUTPUT_DIR' \
                    --num_epochs 2 \
                    --micro_batch_size 8
```
Training on a single Nvidia 4090 GPU is expected to take approximately 180 hours. Upon completion, the fine-tuned model weights will be saved in `$output_dir`.

## DST Inference (`generate_zero_shot.py`, `gpt.py`)

You can load the provided weights directly from the \checkpoint folder and perform inference.

How to make inference with pre-trained model:

```ruby
python3 generate_zero_shot.py --load_8bit True \
                              --base_model 'meta-llama/Meta-Llama-3-8B' \
                              --lora_weights '$OUTPUT_DIR' \
                              --testfile_name '$DATA_DIR' \
                              --testfile_idx '$DATA_DIR' \
                              --output_file '$OUTPUT_DIR'
```

How to make inference with GPT API:

```ruby
python3 generate.py --temperature 0.2 \
                    --test_data_dir '$DATA_DIR' \
                    --test_data_idx '$DATA_DIR' \
                    --output_dir '$OUTPUT_DIR' \
                    --output_file '$OUPUT_DIR/output/'
```

## DST Evaluation

```ruby
python3 eval.py --data_dir '$DATA_DIR' \
                       --output_dir '$DATA_DIR/output/' \
                       --test_idx '$DATA_DIR'
```

## GNN Training

To train the gnn with the VGAE model we created 

```ruby
python3 train_gnn.py --data_dir '$DATA_DIR' \
                     --output_dir '$OUTPUT_DIR'
```



## Acknowledgements

1. [LDST](https://github.com/WoodScene/LDST) : upon which our overall code is built.

They have been a great addition to our research.

