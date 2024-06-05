import os
import json
from glob import glob
import argparse
import openai
import time
import pandas as pd
from tqdm import tqdm

openai.api_key = "api key"


class ChatGPT:
    def __init__(self, conversation_list=[], instruct = "", temperature = 0) -> None:
        self.instruct = instruct
        self.temperature = temperature
        self.conversation_list = [{'role':'system','content': self.instruct}]
        print(f": {self.instruct}\n")
        #self.conversation_list = []
        
    def show_conversation(self,msg_list):
        for msg in msg_list[-2:]:
            if msg['role'] == 'user':
                print(f"(User): {msg['content']}\n")
            else:
                print(f"(System): {msg['content']}\n")

    def retry_request(self, max_retries):
        retries = 0
        while retries < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    #model = "text-davinci-003",
                    messages=self.conversation_list,
                    temperature = self.temperature,

                    )
                return response
            except openai.error.RateLimitError as e:
                print("Request failed:", e)
                print("Retrying in 10 seconds...")
                time.sleep(10)
                retries += 1
        print("Maximum number of retries reached. Request failed.")
        return None
    
    # chatgpt
    def ask(self,prompt):
        self.conversation_list = [{'role':'system','content': self.instruct}]
        self.conversation_list.append({"role":"user","content":prompt})
        
        response = self.retry_request(10)
        answer = response.choices[0].message['content']
        self.conversation_list.append({"role":"assistant","content":answer})
        self.show_conversation(self.conversation_list)
        return answer
        
def operation(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
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
    with open("testfile.txt", 'r') as file:
        for line in file:
            file_list.append(line.strip())

    filter_data = dial_data[dial_data['key'].isin(file_list)]
    filter_key = filter_data.key.unique()
    filter_data.reset_index(drop=True, inplace=True)
    # print(len(filter_data))
    test_data = filter_data[filter_data["response_type"] != "System"]
    # print(len(test_data))

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
        dial_text = eval(test_data['text'][idx_].strip())
        chat = ChatGPT(instruct= args.instruct, temperature=args.temperature)
        request_data = ""

        request_data += """You are a context-aware dialogue specialist, skilled in recognizing user intents and maintaining seamless conversation flow by accurately tracking dialogue states. Below is an instruction that describes a task, paired with an input that provides further context. 
        Write a response that appropriately completes the request. Ensure that the response is clear, concise, and directly addresses the task described in the instruction. 
        Avoid asking for personal information or making assumptions beyond the provided context. Let's think step by step"
        """
        request_data = request_data + dial_text  +"\n"

        print("Send request...")
        answer = chat.ask(request_data)
        result_out.write(dial_text + "|||" + str(answer))
        result_out.write("\n")


    result_out.close()
    print("done.")


def main(args):
    operation(args)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruct", type=str, 
                        default="Answer in the format {‘domain’: , ‘slot’: , ‘value’: }. Track each slot-value if multiple are present in the sentence without any explanation. Choose the domain from ['Hospital', 'Restaurant', 'Hotel', 'Attraction', 'Taxi', 'Train', 'Police', 'Bus', 'General']. Perform Dialogue State Tracking for the given utterance:", help = "")
  
    parser.add_argument("--temperature", type=float, default=0.2, help="")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--output_file", type=str, default="GPT3.5_cot.txt")

    args = parser.parse_args()
    main(args)
