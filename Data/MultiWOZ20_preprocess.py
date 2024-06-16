
import os
import sys
import json
from glob import glob
from random import sample

def preprocess(dial_json, schema, out, idx_out, domain_list, frame_idxs, test_json_list, val_json_list, split):
  dial_json_n = dial_json.split("/")[-1]
  dial_json = open(dial_json)
  dial_json = json.load(dial_json)

  count = 0
  for dial_idx in dial_json.keys():

    if split == "train":
        if dial_idx in test_json_list or dial_idx in val_json_list:
            continue
    elif split == "dev":
        if dial_idx not in val_json_list :
            continue
    elif split == "test":
        if dial_idx not in test_json_list :
            continue
    else:
        print("error")
        sys.exit(1)

    dial = dial_json[dial_idx]['log']
    cur_dial = ""

    for turn_id in range(len(dial)):
      content = dial[turn_id]
      if len(content['metadata']) == 0:
        speaker = " [USER] " 
      else:
        speaker = " [SYSTEM] "   
      uttr = content['text']
      cur_dial += speaker
      cur_dial += uttr  

      if len(content['metadata']) > 0:
        active_slot_values = {}
        turn_slot_dic = content['metadata']
        for frame_key in turn_slot_dic.keys():
          #print(frame_key) #taxi
          if frame_key not in domain_list:
              continue
          for slot_act in turn_slot_dic[frame_key]['semi'].keys():
              if turn_slot_dic[frame_key]['semi'][slot_act] != "":
                active_slot_values[frame_key.lower()+"-"+slot_act.lower()] = turn_slot_dic[frame_key]['semi'][slot_act] # 这也不一样
          for slot_act in turn_slot_dic[frame_key]['book'].keys():
              if slot_act != "booked" and turn_slot_dic[frame_key]['book'][slot_act] != "":
                active_slot_values[frame_key.lower()+"-"+slot_act.lower()] = turn_slot_dic[frame_key]['book'][slot_act] # 这也不一样n 
        for domain in schema.keys():

          slots = [domain.split("-")[1]]
          d_name = domain.split("-")[0]
          for slot in slots:
            s_name = slot
            schema_prompt = ""
            schema_prompt += " [domain] " + d_name + ","
            schema_prompt += " [slot] " + s_name + "."
            if PVs_flag:
              if len(schema[domain]) > 0:
                PVs = ", ".join(schema[domain])
                schema_prompt += " [Possible Values] " + PVs
            domain_slot = d_name.lower() + "-" + s_name.lower()
            s_name2 = s_name
            while " " in s_name2:
                s_name2 = s_name2.replace(" ", "")
            domain_slot2 = d_name.lower() + "-" + s_name2.lower()
            if domain_slot in active_slot_values.keys():
              target_value = active_slot_values[domain_slot]
            elif domain_slot2 in active_slot_values.keys():
              target_value = active_slot_values[domain_slot2]
            else:
              target_value = "NONE"
            
            line = { "dialogue": cur_dial + schema_prompt, "state":  target_value }

            out.write(json.dumps(line))
            out.write("\n")
            idx_list = [ dial_json_n, str(dial_idx), str(turn_id), str(frame_idxs[d_name]), d_name, s_name ]
            idx_out.write("|||".join(idx_list))
            idx_out.write("\n")
      count +=1
  return


def Generate_data(domain_list):
    data_path = "./MULTIWOZ20/"
    #data_path = sys.argv[1]

    parent_dir = os.path.dirname(os.path.dirname(data_path))
    multiwoz_dir = os.path.basename(os.path.dirname(data_path))
    data_path_out = os.path.join(parent_dir, multiwoz_dir+"_preprocess")
    if not os.path.exists(data_path_out):
      os.makedirs(data_path_out)
    print(data_path_out)
    #sys.exit(1)

    frame_idxs = {}
    for service_idx in range(len(domain_list)):
        service = domain_list[service_idx]
        frame_idxs[service] = service_idx
  
    # skip domains that are not in the testing set
    #excluded_domains = ["police", "hospital", "bus"]
    
    test_list_file = open(os.path.join(data_path, "{}.json".format("testListFile")))
    test_json_list = []
    for lin in test_list_file:
        test_json_list.append(lin.strip())
    
    val_list_file = open(os.path.join(data_path, "{}.json".format("valListFile")))
    val_json_list = []
    for lin in val_list_file:
        val_json_list.append(lin.strip())
    #print(len(val_json_list))
    #sys.exit()
    
    for split in ["train","dev", "test"]:
    #for split in [ "test"]:
        print("--------Preprocessing {} set---------".format(split))
        out = open(os.path.join(data_path_out, "{}.json".format(split)), "w") # W 模式 会覆盖之前的内容
        idx_out = open(os.path.join(data_path_out, "{}.idx".format(split)), "w")
        dial_jsons = glob(os.path.join(data_path, "*json"))
        
        schema_path = data_path + "ontology.json"
        schema = json.load(open(schema_path))

        for dial_json in dial_jsons:
            if "data.json" in dial_json:
                preprocess(dial_json, schema, out, idx_out, domain_list, frame_idxs, test_json_list, val_json_list, split)
        idx_out.close()
        out.close()
    print("--------Finish Preprocessing---------")

def Analysis():
    data_path = "./MULTIWOZ20/"

    schema_path = data_path + "ontology.json"
    schema = json.load(open(schema_path))


    domain_list = []
    slot_list = []

    for domain in schema.keys():
        #print(domain['service_name'])
        d = domain.split("-")[0]
        slot_list.append(domain)
        if d not in domain_list:
            domain_list.append(d)
   
    print(domain_list)
       
    print(slot_list)

    print(len(domain_list))



    return domain_list

def main():
    domain_list = Analysis()
    print(domain_list)
    Generate_data(domain_list)

if __name__=='__main__':
    main()
