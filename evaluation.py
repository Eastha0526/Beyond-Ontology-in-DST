import fire
import json
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_slot_extraction(data, dialog_id, turn_index, domain, slot, predicted_value):
    if dialog_id not in data:
        return False, "NONE"
    
    dialog = data[dialog_id]
    log = dialog["log"]
    
    if turn_index >= len(log):
        return False, "NONE"
    
    turn = log[turn_index]
    if "metadata" not in turn:
        return False, "NONE"
    
    if domain not in turn["metadata"]:
        return False, "NONE"
    
    if len(parts[5].split(" ")) > 1:
        intent = parts[5].split(" ")[0]
        slot = parts[5].split(" ")[1]
        actual_value = turn["metadata"][domain][intent].get(slot)
    else:
        actual_value = turn["metadata"][domain].get(slot)

    if actual_value == "None":
        actual_value = "NONE"
    elif actual_value == "":
        actual_value = "NONE"
    elif actual_value is None:
        actual_value = "NONE"

    actual_value = actual_value.upper()
    predicted_value = predicted_value.upper()

    print(f" act {actual_value} | pred : {predicted_value}")

    return actual_value == predicted_value, actual_value

def make_evaluation():

    data_file_path: str = "",
    result_file_path: str = "",
    output_file_path: str = "",

    data_file_path = "data.json"
    with open(data_file_path, 'r') as f:
        data = json.load(f)

    result_file_path = "llama_result_multiwoz20.txt"
    with open(result_file_path, 'r') as f:
        lines = f.readlines()

    evaluation_data = []
    for line in lines:
        parts = line.strip().split("|||")
        data_path = parts[0] 
        dialog_id = parts[1] 
        turn_index = int(parts[2]) 
        domain_index = int(parts[3]) 
        domain = parts[4]
        slot = parts[5] 
        predicted_value = parts[6] 
        evaluation_data.append((dialog_id, turn_index, domain_index, domain, slot, predicted_value))

    results = []
    true_values = []
    pred_values = []
    turn_correct = 0
    total_turns = len(evaluation_data)
    for dialog_id, turn_index, domain_index, domain, slot, predicted_value in evaluation_data:
        is_correct, actual_value = evaluate_slot_extraction(data, dialog_id, turn_index, domain, slot, predicted_value)
        result_line = f"{dialog_id}|||{turn_index}|||{domain_index}|||{domain}|||{slot}|||{predicted_value}|||{is_correct}|||{actual_value}\n"
        results.append(result_line)
        
        if is_correct:
            turn_correct += 1
            
        true_values.append(actual_value)
        pred_values.append(predicted_value)

    joint_accuracy = turn_correct / total_turns

    output_file_path = ""
    with open(output_file_path, 'w') as f:
        f.writelines(results)

    print(f"Joint Accuracy: {joint_accuracy}")


if __name__ == '__main__':
    fire.Fire(make_evaluation)


    