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
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    lora_weights: str = "./Checkpoint_files/checkpoint-2200",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    split_id: str = "",
    testfile_name: str = "./Data/MULTIWOZ2.4_preprocess/sample.json",
    testfile_idx: str = "./Data/MULTIWOZ2.4_preprocess/sample.idx",
    output_file: str = "./0520.txt",
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

    def evaluate(
        input=None,
        temperature=0.02,
        top_p=0,
        top_k=1,
        num_beams=1,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(input)
        print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,

            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }
        
        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.pad_token_id,
                bos_token_id=model.config.bos_token_id,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)
        #return tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        #return output.split("### Response:")[1].strip()



    # testing code for readme
    # for instruction in [
    #     "Tell me about alpacas.",
    #     "Tell me about the president of Mexico in 2019.",
    #     "Tell me about the king of France in 2019.",
    #     "List all Canadian provinces in alphabetical order.",
    #     "Write a Python program that prints the first 10 Fibonacci numbers.",
    #     "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
    #     "Tell me five words that rhyme with 'shock'.",
    #     "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    #     "Count up from 1 to 500.",
    # ]:
    #     print("Instruction:", instruction)
    #     print("Response:", evaluate(instruction))
    #     print()

    def read_json_as_list(filename):
        data_list = []
        with open(filename, 'r') as file:
            # 파일 내용 전체를 읽습니다.
            content = file.read()
            # 각 JSON 객체를 분리합니다.
            # 표준적인 형태가 아니므로, 정확한 구분 방법을 사용해야 합니다.
            # 예를 들어, '}'와 '{' 사이에 새 줄이 있다고 가정하면 다음과 같이 할 수 있습니다:
            objects = content.split('}\n{')
            # 첫 번째와 마지막 객체를 정리합니다.
            objects[0] = objects[0] + '}'
            objects[-1] = '{' + objects[-1]
            # 중간 객체들에 대해서는 앞뒤로 중괄호를 추가해줍니다.
            for i in range(1, len(objects)-1):
                objects[i] = '{' + objects[i] + '}'
            # 각 객체를 JSON으로 파싱하여 리스트에 추가합니다.
            for obj in objects:
                data_list.append(json.loads(obj))
        return data_list


    print(f"output filename: {output_file}")
    print(f"test filename: {testfile_name}")
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
    
    idx_lines = open(testfile_idx).readlines()
  
    data = read_json_as_list(testfile_name)
    for idx_ in tqdm(range(begin_id, len(data))):
    #for idx_ in range(begin_id, len(data)):
        sample = data[idx_]
        idx_line = idx_lines[idx_].strip()
        # print(sample)
        dial_json_n, dial_idx, turn_idx, frame_idx, d_name, s_name = idx_line.split("|||") 

        Response_list = []
        input_initial = sample['dialogue']

        idx1 = input_initial.find("[domain]")
        idx2 = input_initial.find("So the value of slot")
        input2 = input_initial[:idx1] + "\n" + input_initial[idx2:]
        input2 = input2 + "If the slot is not mentioned in the dialogue, just return NONE. \n "
        input2 = input2 + "So the value of slot <"+ d_name+ "-" + s_name +"> is "
        print(input2)
        Response = evaluate(input = input2 + "\n")
        Response_list.append(Response)

        #print("Input:", input2)
        print("Response list:", Response_list)
        print("Ground truth:", sample['state'])
        print()

        result_out.write(idx_line + "|||" + str(Response_list))
        result_out.write("\n")

        #break
    result_out.close()

if __name__ == "__main__":
    fire.Fire(main)