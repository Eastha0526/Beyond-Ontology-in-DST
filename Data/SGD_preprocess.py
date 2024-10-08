

import os
import sys
import json
from glob import glob
from random import sample

def preprocess(dial_json, schema, out, idx_out, same_service_list, frame_idxs):
  dial_json_n = dial_json.split("/")[-1]
  dial_json = open(dial_json)
  dial_json = json.load(dial_json)
  count = 0
  for dial_idx in range(len(dial_json)):

    dial = dial_json[dial_idx]
    dial_services = dial['services']
    count_temp = 0
    for dial_service in dial_services:
        if dial_service in same_service_list:
            break
        count_temp +=1 
    if count_temp == len(dial_services):
        continue
    
    turn_id = 0
    for turn in dial["turns"]:
      speaker = " [" + turn["speaker"] + "] " 
      uttr = turn["utterance"]
      cur_dial += speaker
      cur_dial += uttr  

      if turn["speaker"] == "USER":
        active_slot_values = {}
        for frame_idx in range(len(turn["frames"])):
          frame = turn["frames"][frame_idx]

          for key, values in frame["state"]["slot_values"].items():
            value = sample(values,1)[0]
            active_slot_values[frame['service']+"-"+key] = value # 这也不一样
        for domain in schema:
          # skip domains that are not in the testing set
          if domain["service_name"] not in same_service_list:
            continue
          slots = domain["slots"]
          d_name = domain["service_name"]
          for slot in slots:
            s_name = slot["name"]
            schema_prompt = ""
            schema_prompt += " [domain] " + d_name + ", it indicates " + domain["description"] if domain_desc_flag else d_name
            schema_prompt += " [slot] " + s_name + ", it indicates " + slot["description"] if slot_desc_flag  else s_name
            if PVs_flag:
              if slot["is_categorical"]:
                PVs = ", ".join(slot["possible_values"])
                schema_prompt += " [Possible Values] " + PVs
            if domain_slot in active_slot_values.keys():
              target_value = active_slot_values[domain_slot]
            else:
              target_value = "NONE"
            
            line = { "dialogue": cur_dial + schema_prompt, "state":  target_value }

            out.write(json.dumps(line))
            out.write("\n")

            idx_list = [ dial_json_n, str(dial_idx), str(turn_id), str(frame_idxs[d_name]), d_name, s_name ]
            idx_out.write("|||".join(idx_list))
            idx_out.write("\n")
      count +=1
      turn_id += 1
  return


def Generate_data(same_service_list):
    data_path = "./SGD/"
    #data_path = sys.argv[1]
    parent_dir = os.path.dirname(os.path.dirname(data_path))
    multiwoz_dir = os.path.basename(os.path.dirname(data_path))
    data_path_out = os.path.join(parent_dir, multiwoz_dir+"_preprocess")
    if not os.path.exists(data_path_out):
      os.makedirs(data_path_out)
    print(data_path_out)
    #sys.exit(1)

    frame_idxs = {}
    for service_idx in range(len(same_service_list)):
        service = same_service_list[service_idx]
        frame_idxs[service] = service_idx
    
    for split in ["train","dev", "test"]:
    #for split in [ "test"]:
        print("--------Preprocessing {} set---------".format(split))
        out = open(os.path.join(data_path_out, "{}.json".format(split)), "w")
        idx_out = open(os.path.join(data_path_out, "{}.idx".format(split)), "w")
        dial_jsons = glob(os.path.join(data_path, "{}/*json".format(split)))
        
        schema_path = data_path + split + "/schema.json"
        schema = json.load(open(schema_path))

        for dial_json in dial_jsons:
            if dial_json.split("/")[-1] != "schema.json" and "schema.json" not in dial_json:
                print(dial_json)
                preprocess(dial_json, schema, out, idx_out, same_service_list, frame_idxs)
        idx_out.close()
        out.close()
    print("--------Finish Preprocessing---------")

def Analysis():
    data_path = "./SGD/"

    schema_path = data_path + "train/schema.json"
    schema_train = json.load(open(schema_path))

    schema_path = data_path + "test/schema.json"
    schema_test = json.load(open(schema_path))
    
    domain_train_list = []
    train_dic = {}
    domain_test_list = []
    test_dic = {}
    
    same_service_list = []
    zero_shot_service_list = []
    count = 0
    for domain in schema_train:
        #print(domain['service_name'])
        domain_train_list.append(domain['service_name'])
        if domain['service_name'].split("_")[0] not in train_dic:
          train_dic[domain['service_name'].split("_")[0]] = 1
        else:
          train_dic[domain['service_name'].split("_")[0]] += 1
          
    for domain in schema_test:
        #print(domain['service_name'])
        if domain['service_name'] in domain_train_list:
            count += 1
            same_service_list.append(domain['service_name'])
        
        domain_test_list.append(domain['service_name'])
        if domain['service_name'].split("_")[0] not in test_dic:
          test_dic[domain['service_name'].split("_")[0]] = 1
        else:
          test_dic[domain['service_name'].split("_")[0]] += 1
        if domain['service_name'].split("_")[0] not in train_dic.keys():
            zero_shot_service_list.append(domain['service_name'])
        
    return same_service_list

def main():
    same_service_list = Analysis()
    print(same_service_list)
    Generate_data(same_service_list)

if __name__=='__main__':
    main()
