# prompt type 1 and 3 in Fig 2.
# fix prompt, add: "simply based on the content of the dialogue"

import os
import json
import copy
from glob import glob
import argparse
import openai
import time
import sys
import re
import pandas as pd
from tqdm import tqdm


from openai import OpenAI

# Edit this part for your setup
client = OpenAI(api_key="sk-proj-iKFnug5HFGu9KocuEpmkT3BlbkFJZ7doaLBhNoaejeFQa09N")
# client = OpenAI(api_key="local-generated-key", base_url="http://devnuc.lan:5000/v1")

def query_llm(messages, max_tokens=2048, temperature=0.1):
    # Retry forever
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
            )

            content = response.choices[0].message.content.strip()

            return content
        except Exception as e:
            print("Failure querying the AI. Retrying...")
            time.sleep(1)

def query_openai(prompt):
    messages = [
        { "role": "user", "content": prompt }
    ]
    return query_llm(messages)


# STAGE 1

def select_reasoning_modules(task_description, reasoning_modules):
    """
    Step 1: SELECT relevant reasoning modules for the task.
    """
    prompt = f"Given the task: {task_description}, which of the following reasoning modules are relevant? Do not elaborate on why.\n\n" + "\n".join(reasoning_modules)
    selected_modules = query_openai(prompt)
    return selected_modules

def adapt_reasoning_modules(selected_modules, task_example):
    """
    Step 2: ADAPT the selected reasoning modules to be more specific to the task.
    """
    prompt = f"Without working out the full solution, adapt the following reasoning modules to be specific to our task:\n{selected_modules}\n\nOur task:\n{task_example}"
    adapted_modules = query_openai(prompt)
    return adapted_modules

def implement_reasoning_structure(adapted_modules, task_description):
    """
    Step 3: IMPLEMENT the adapted reasoning modules into an actionable reasoning structure.
    """
    prompt = f"Without working out the full solution, create an actionable reasoning structure for the task using these adapted reasoning modules:\n{adapted_modules}\n\nTask Description:\n{task_description}"
    reasoning_structure = query_openai(prompt)
    return reasoning_structure

# STAGE 2

def execute_reasoning_structure(reasoning_structure, task_instance):
    """
    Execute the reasoning structure to solve a specific task instance.
    """
    prompt = f"Using the following reasoning structure: {reasoning_structure}\n\nSolve this task, providing your final answer: {task_instance}"
    solution = query_openai(prompt)
    return solution



def operation(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    filter_data = pd.read_csv("./sample_jga.csv")
    filter_data.reset_index(drop=True, inplace=True)
    print(len(filter_data))
    test_data = filter_data[filter_data["turn_type"] != "SYSTEM"]
    test_data.reset_index(drop=True, inplace=True)
    print(len(test_data))

    output_file = os.path.join(args.output_dir, args.output_file)

    if not os.path.isfile(output_file): 
        result_out = open(output_file, "w", encoding='utf-8')
        begin_id = 0 
    else: 
        with open(output_file, "r") as f:
            lines = f.readlines()
            begin_id = len(lines)
            f.close()
        result_out = open(output_file, "a", encoding='utf-8')
    
    for idx_ in tqdm(range(begin_id, len(test_data)), desc="Processing Dialogues"):
        
        dial_text = test_data['utterance_list'][idx_]
        domain = test_data["domain_list"][idx_]
        slot = test_data["domain_list"][idx_]

        
        request_data = """You are a context-aware dialogue specialist, skilled in recognizing user intents and maintaining seamless conversation flow by accurately tracking dialogue states. Below is an instruction that describes a task, paired with an input that provides further context. 
        Write a response that appropriately completes the request. Ensure that the response is clear, concise, and directly addresses the task described in the instruction. 
        Avoid asking for personal information or making assumptions beyond the provided context."
        """
        request_data = request_data + dial_text  +"\n"
        reasoning_modules = "Let’s think  step by step."
        instruction = "Answer in the format {‘domain’: , ‘slot’: , ‘value’: }. Track each slot-value if multiple are present in the sentence without any explanation. If value doesn't exist, return value as “NONE”. Choose the domain from ['Hospital', 'Restaurant', 'Hotel', 'Attraction', 'Taxi', 'Train', 'Police', 'Bus']. Perform Dialogue State Tracking for the given utterance:"

        task_description = instruction + dial_text + "domain is" + domain + "slot is" + slot

        selected_modules = select_reasoning_modules(task_description, reasoning_modules)
        print("Stage 1 SELECT: Selected Modules:\n", selected_modules)

        adapted_modules = adapt_reasoning_modules(selected_modules, task_description)
        print("\nStage 1 ADAPT: Adapted Modules:\n", adapted_modules)

        reasoning_structure = implement_reasoning_structure(adapted_modules, task_description)
        print("\nStage 1 IMPLEMENT: Reasoning Structure:\n", reasoning_structure)

        result = execute_reasoning_structure(reasoning_structure, task_description)
        print("\nStage 2: Final Result:\n", result)

        result_out.write(dial_text + "|||" + str(result))
        
    result_out.close()
    print("done.")



def main(args):
    operation(args)
    
    
if __name__=='__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument("--instruct", type=str, 
    #                     default="Now you need to perform the task of multi-domain dialogue state tracking. And I will show you an example and you need to return to the state of the slot I asked about.", help = "")
    # parser.add_argument("--instruct", type=str, 
    #                     default="Now you need to perform the task of multi-domain dialogue state tracking. And I will show you an example and you need to return the answer strictly in the format of the example.", help = "")
    # parser.add_argument("--instruct", type=str, 
    #                     default="Now you need to perform the task of multi-domain dialogue state tracking. You need to return the value of the slot I'm asking about simply based on the content of the dialogue. And I will provide you with an example.", help = "")
    parser.add_argument("--instruct", type=str, 
                        default="Answer in the format {‘domain’: , ‘slot’: , ‘value’: }. Track each slot-value if multiple are present in the sentence without any explanation. Choose the domain from ['Hospital', 'Restaurant', 'Hotel', 'Attraction', 'Taxi', 'Train', 'Police', 'Bus', 'General']. Perform Dialogue State Tracking for the given utterance:", help = "")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--output_file", type=str, default="GPT3.5_self_sgd.txt")

    args = parser.parse_args()
    main(args)
