import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import argparse

def eval(eval_data, result_data, output_results):

    df = pd.read_csv(eval_data)
    with open(result_data, 'r', encoding='utf-8') as file:
        content = file.read()

    text = []
    result = []
    parts = content.split('|||')

    for part in parts[1:]:
        first_match = part[part.find("### Input:"):]
        first_results = first_match[first_match.find("\n")+1:]
        match = part[part.find("### Response:"):]
        second_match = match[match.find("\n")+1:]
        text.append(first_results[:first_results.find("Let's")-1])
        result.append(second_match[:second_match.find("\n")])

    data = pd.DataFrame({"text" : text, "result": result})

    def extract_value(input_string):
        pattern = r"‘value’: ‘([^’]+)’"
        matches = re.findall(pattern, input_string)
        return matches

    def extract_domain(input_string):
        pattern = r"‘domain’: ‘([^’]+)’"
        matches = re.findall(pattern, input_string)
        return matches

    def extract_slot(input_string):
        pattern = r"'slot':\s*'([^']+)'"
        matches = re.findall(pattern, input_string)
        return matches
    
    eval["value_pred"] = eval["result"].apply(extract_value)
    eval["pred_domain"] = eval["result"].apply(extract_domain)
    le = LabelEncoder()
    eval["number"] = le.fit_transform(eval["text"])

    true_labels_all = []
    pred_labels_all = []

    domain_results = {}
    df_cleaned = df[df['value'].astype(bool)]
    df_cleaned.dropna(inplace=True)
    df_cleaned["value"] = df_cleaned["value"].apply(lambda x: "NONE" if x == "[]" else x)

    for index, row in df_cleaned.iterrows():
        domain = row['domain']
        if domain not in domain_results:
            domain_results[domain] = {'true_labels': [], 'pred_labels': [], 'count': 0}

        if row['domain'] in row['pred_domain']:
            if row['value'] == 'yes':
                pred = any(vp == 'yes' for vp in row['value_pred'])
                true = True
            else:
                pred = row['value'] in row['value_pred']
                true = True
        else:
            pred = False
            true = False

        domain_results[domain]['true_labels'].append(1 if true else 0)
        domain_results[domain]['pred_labels'].append(1 if pred else 0)
        domain_results[domain]['count'] += 1

        true_labels_all.append(1 if true else 0)
        pred_labels_all.append(1 if pred else 0)

    results = []

    for domain, labels in domain_results.items():
        precision = precision_score(labels['true_labels'], labels['pred_labels'])
        recall = recall_score(labels['true_labels'], labels['pred_labels'])
        f1 = f1_score(labels['true_labels'], labels['pred_labels'])
        accuracy = accuracy_score(labels['true_labels'], labels['pred_labels'])
        count = labels['count']

        results.append([domain, count, f1, accuracy])

    precision_all = precision_score(true_labels_all, pred_labels_all)
    recall_all = recall_score(true_labels_all, pred_labels_all)
    f1_all = f1_score(true_labels_all, pred_labels_all)
    accuracy_all = accuracy_score(true_labels_all, pred_labels_all)

    results.append(['Overall', len(df), f1_all, accuracy_all])

    results_df = pd.DataFrame(results, columns=['Domain', 'Count', 'F1 Score', 'Accuracy'])
    print(results_df)
    results_df.to_csv(output_results, sep='\t', index=False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process some data files.")
    parser.add_argument('--eval_data', type=str, required=True, help="Path to the evaluation data CSV file")
    parser.add_argument('--result_data', type=str, required=True, help="Path to the result data text file")
    parser.add_argument('--output_results', type=str, required=True, help="Path to the output results")

    args = parser.parse_args()
    eval(args.eval_data, args.result_data, args.output_results)