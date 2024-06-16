
import random
import json
import sys

def Template1(dial_text):
    dial = dial_text['dialogue'].split(" [domain] ")
    input1 = ""
    input1 = input1 + dial[0] + " \n "
    return input1, output1


def Template2(dial_text):
    dial = dial_text['dialogue'].split(" [domain] ")
    input1 = ""
    input1 = input1 + dial[0] + " \n"
    return input1, output1
  

if __name__ == '__main__':
    activate_number = 475217
    
    except_number = 0
    original_number = 0
    for data_type in ["train"]:
        data_dir = "./MULTIWOZ2.4_preprocess/"+ data_type +".json"
        data_idx = "./MULTIWOZ2.4_preprocess/"+ data_type +".idx"
        
        output_filename = "./MULTIWOZ2.4_preprocess/"+ data_type +"_LLM_zero-shot_except-"+ str(except_domain) +"-domain.json"
        dataset_data = []
        
        idx_lines = open(data_idx).readlines()
        test_data_lines = open(data_dir).readlines()
        
        assert len(idx_lines) == len(test_data_lines)
        
        none_sample_number = len(idx_lines) - activate_number

        for idx_ in range(len(idx_lines)):
            dial_text = eval(test_data_lines[idx_].strip())
            #print(dial_text['dialogue'])
            if dial_text['state'] == "NONE":
                if random.random() > none_sample_select_ratio:
                    continue
            
            original_number += 1
            item = {}
            idx_list = idx_lines[idx_].strip()
            dial_json_n, dial_idx, turn_idx, frame_idx, d_name, s_name = idx_list.split("|||") 

            if d_name == except_domain:
                except_number +=1 
                continue

            if random.random() > 0.1:
                instru = "Track the state of the slot in the input dialogue."
            else:
                instru = "Track the state of the slot <"+ d_name+ "-" + s_name +"> in the input dialogue."
            if template_id == 1:
                input1, output1 = Template1(dial_text)
            elif template_id == 2:
                input1, output1 = Template2(dial_text)
            elif template_id == 3:
                input1, output1 = Template3(dial_text)
            else:
                print("error")
                sys.exit(1)
            
            item['instruction'] = instru
            item['input'] = input1
            dataset_data.append(item)
        
        with open(output_filename, 'w') as f:
            json.dump(dataset_data, f, indent=4)    
