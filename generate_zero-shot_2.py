import os
import sys
import json
import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
import pandas as pd

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = True,
    base_model: str = "meta-llama/Meta-Llama-3-8B",
    lora_weights: str = "./Checkpoint_files/checkpoint-3200",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    split_id: str = "",
    output_file: str = "./0609_multiwoz.txt",
    except_domain: str = "",
):
    
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    print(f"lora_weights: {lora_weights}")
    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    #print(torch.cuda.is_available())
    #sys.exit(1)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
    #print(model.config.use_cache)
    #sys.exit(1)
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    instruction = "Answer in the format {'domain': , 'slot': , 'value': }. Track each slot-value if multiple are present in the sentence without any explanation. If value doesn't exist, return value as “NONE”. Choose the domain from ['Hospital', 'Restaurant', 'Hotel', 'Attraction', 'Taxi', 'Train', 'Police', 'General']. Perform Dialogue State Tracking for the given utterance by identifying the domain, slot, and value for each relevant piece of information provided in the user's input. Consider the domain when identifying the slot and value. Perform Dialogue State Tracking for the given utterance:"
    def get_intent(context):

        prompt = prompter.generate_prompt(instruction, context)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0,
            top_k=1,
            num_beams=1,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=128,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.pad_token_id,
                bos_token_id=model.config.bos_token_id,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)


    output_folder = os.path.dirname(output_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    if not os.path.isfile(output_file): 
        result_out = open(output_file, "w", encoding='utf-8')
        begin_id = 0 # 
        print("——————————————————————————————Write from scratch——————————————————————————————")
    else: # 
        with open(output_file, "r") as f:
            lines = f.readlines()
            begin_id = len(lines)
            f.close()
        print(f"——————————————————————————————Write from line {begin_id}——————————————————————————————")
        result_out = open(output_file, "a", encoding='utf-8')
    

    with open("data.json") as f:
        data = json.load(f)
    
    dial_key = data.keys()

    key_list = []
    text_list = []
    turn_list = []
    user_list = []

    for key in dial_key:
        cnt = 0
        for text in (data[key]['log']):
            key_list.append(key)
            turn_list.append(cnt)
            text_list.append(text['text'])
            cnt += 1

            if cnt & 1 == 1:
                user_list.append("User")
            else:
                user_list.append("System")

    dial_data = pd.DataFrame({"key" : key_list,"turn" : turn_list ,"text" : text_list, "response_type" : user_list})
    dial_data["domain_type"] = dial_data['key'].apply(lambda x: "multi" if 'MUL' in str(x) else "single")


    file_list = []
    with open("testFile.txt", 'r') as file:
        for line in file:
            file_list.append(line.strip())

    filter_data = dial_data[dial_data['key'].isin(file_list)]
    filter_key = filter_data.key.unique()
    filter_data.reset_index(drop=True, inplace=True)
    print(len(filter_data))
    filter_data = filter_data[filter_data["response_type"] != "System"]
    print(len(filter_data))

    for idx, data in (filter_data[filter_data['key'] == key].iterrows()):
#for idx_ in range(begin_id, len(data)):
        context = data['text']
        print(context)
        context = str(context)
        intent = get_intent(context)
        print(intent)
        result_out.write(context + "|||" + str(intent))
        result_out.write("\n")
        #break
    result_out.close()

if __name__ == "__main__":
    fire.Fire(main)
