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
from tqdm import tqdm
openai.api_key = "sk-proj-iKFnug5HFGu9KocuEpmkT3BlbkFJZ7doaLBhNoaejeFQa09N"


class ChatGPT:
    def __init__(self,conversation_list=[], instruct = "", temperature = 0) -> None:
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
                    model="gpt-4o",
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
        
    idx_lines = open(args.test_data_idx).readlines()
    #test_data = json.load(open(args.test_data_dir))
    test_data = open(args.test_data_dir).readlines()
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
        dial_text = eval(test_data[idx_].strip())
        idx_line = idx_lines[idx_].strip()
        
        dial_json_n, dial_idx, turn_idx, frame_idx, d_name, s_name = idx_line.split("|||") 
        
        #Please return the value of slot: <hotel-pricerange>. It means price budget of the hotel. It was categorical, and you should select the value in the possible value list: ['expensive', 'cheap', 'moderate']. If the slot is not mentioned in the current dialogue, just return NONE. So the value of slot <hotel-pricerange> is
        dial = dial_text['dialogue'].split(" [domain] ")
        input1 = ""
        input1 = input1 + dial[0] + " \n\n "
        
        if "Possible Values" in dial[1]:
            dial2 = dial[1].split(" [Possible Values] ")
            input1 = input1 + "[domain] " + dial2[0] + ". "
            input1 = input1 + "This slot is categorical and you can only choose from the following available values: "
            input1 = input1 + dial2[1] + ". "
        else:
            input1 = input1 + "[domain] " + dial[1] + ". "

        input1 = input1 + "If the slot is not mentioned in the dialogue, just return NONE. \n "

        input1 = input1 + "So the value of slot <"+ d_name+ "-" + s_name +"> is "


        chat = ChatGPT(instruct= args.instruct, temperature=args.temperature)
        request_data = ""
        # add a demonstration （prompt type3）
        # request_data += "The example is: Input dialogue: [User]: I need train reservations from norwich to cambridge [SYSTEM]: I have 133 trains matching your request. Is there a specific day and time you would like to travel? [User]: I'd like to leave on Monday and arrive by 18:00. [SYSTEM]: There are 12 trains for the day and time you request. Would you like to book it now? [User]: Before booking, I would also like to know the travel time, price, and departure time please. [SYSTEM]: There are 12 trains meeting your needs with the first leaving at 05:16 and the last one leaving at 16:16. Do you want to book one of these? [User]: No hold off on booking for now. Can you help me find an attraction called cineworld cinema? [SYSTEM]: Yes it is a cinema located in the south part of town what information would you like on it?\n"
        # request_data += "So the value of slot <train-departure> is \n"
        # request_data += "And your result should be Norwich.\n\n"

        request_data += "The following is the dialogue you need to test:\n"
        request_data += "Input dialogue: "
        request_data = request_data + input1  +"\n"
        
        #print(request_data)

        # another prompt template
        # request_data = ""
        # request_data += "The example is: Input dialogue: [User]: I need train reservations from norwich to cambridge [SYSTEM]: I have 133 trains matching your request. Is there a specific day and time you would like to travel? [User]: I'd like to leave on Monday and arrive by 18:00. [SYSTEM]: There are 12 trains for the day and time you request. Would you like to book it now? [User]: Before booking, I would also like to know the travel time, price, and departure time please. [SYSTEM]: There are 12 trains meeting your needs with the first leaving at 05:16 and the last one leaving at 16:16. Do you want to book one of these? [User]: No hold off on booking for now. Can you help me find an attraction called cineworld cinema? [SYSTEM]: Yes it is a cinema located in the south part of town what information would you like on it?\n"
        # request_data += "Output result: Train-Departure: Norwich, Train-Arrival: Cambridge, Train-Departure-Time: Monday, Train-Arrival-Time: 18:00, Attraction-Name: cineworld cinema \n\n"
        # request_data += "And you need test this example:\n"
        # request_data += "Input dialogue: "
        # request_data = request_data + dialog_text  +"\n"

        print("Send request...")
        answer = chat.ask(request_data)
        result_out.write(idx_line + "|||" + str(answer))
        result_out.write("\n")


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
                        default="Now you need to perform the task of multi-domain dialogue state tracking. You need to return the value of the slot I'm asking about simply based on the content of the dialogue. No explanation!", help = "")
  
    parser.add_argument("--temperature", type=float, default=0.2, help="")

    parser.add_argument("--test_data_dir", type=str, default="./Data/MULTIWOZ2.4_preprocess/sample.json", help = "")
    parser.add_argument("--test_data_idx", type=str, default="./Data/MULTIWOZ2.4_preprocess/sample.idx", help = "")

    parser.add_argument("--output_dir", type=str, default="./ATOM_result")
    parser.add_argument("--output_file", type=str, default="GPT4o_Result_MultiWOZ24.txt")

    args = parser.parse_args()
    main(args)
