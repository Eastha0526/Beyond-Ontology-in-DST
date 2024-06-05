import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from sklearn.preprocessing import LabelEncoder
from models.VGAE import create_model
from utils.gnn_utils import train, test, create_optimizer, predict
import fire

def main():

    graph_df = pd.read_csv("graph_df.csv")
    domain_encoder = LabelEncoder()
    value_encoder = LabelEncoder()
    graph_df["domain_encoded"] = domain_encoder.fit_transform(graph_df["normalized_domain"])
    graph_df["value_encoded"] = value_encoder.fit_transform(graph_df["normalized_value"])
    unique_domains = graph_df['domain_encoded'].unique()
    unique_values = graph_df['value_encoded'].unique()

    domain_map = {v: i for i, v in enumerate(unique_domains)}
    value_map = {v: i + len(unique_domains) for i, v in enumerate(unique_values)}

    graph_df['domain_mapped'] = graph_df['domain_encoded'].map(domain_map)
    graph_df['value_mapped'] = graph_df['value_encoded'].map(value_map)

    edge_index = torch.tensor([graph_df['domain_mapped'].values, graph_df['value_mapped'].values], dtype=torch.long)

    num_nodes = len(unique_domains) + len(unique_values)
    x = torch.eye(num_nodes)

    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = x.size(0)

    transform = RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, split_labels=True, add_negative_train_samples=True)
    train_data, val_data, test_data = transform(data)

    # print(f"Number of nodes: {data.num_nodes}")
    # print(f"Edge index shape: {data.edge_index.shape}")
    # print(f"Edge index:\n{data.edge_index}")

    model = create_model(data.num_features)
    
    # Create optimizer
    optimizer_name = 'adam'  # You can change this to 'sgd', 'adamw', 'rmsprop', etc.
    optimizer = create_optimizer(model, optimizer_name, lr=0.01)

    # Train and evaluate the model
    for epoch in range(301):
        loss = train(model, optimizer, train_data)
        val_auc, val_ap = test(model, val_data)
        if epoch % 50 == 0:
            print(f"Epoch {epoch:>2} | Loss : {loss:.4f} | Val AUC: {val_auc:.4f} | Val AP : {val_ap:.4f}")

    test_auc, test_ap = test(model, test_data)
    print(f"Test AUC : {test_auc:.4f} | Test AP : {test_ap:.4f}")

    predicted_probs = predict(model, test_data.x, test_data.edge_index)


    count = 0
    num_nodes = test_data.num_nodes
    all_node_pairs = torch.combinations(torch.arange(num_nodes), r=2).t()
    for i in torch.argsort(predicted_probs, descending=True):
        if count >= 10:
            break
        
        node_pair = all_node_pairs[:, i]
        prob = predicted_probs[i].item()
        
        if node_pair[0] <= 7:
            try:
                domain_node = domain_encoder.inverse_transform([int(node_pair[0])])
                domain_label = domain_node[0]
            except ValueError:
                domain_label = f'Unseen domain label: {int(node_pair[0])}'
            except IndexError:
                domain_label = f'Invalid index for domain label: {int(node_pair[0])}'

            try:
                value_node = value_encoder.inverse_transform([int(node_pair[1])])
                value_label = value_node[0]
            except ValueError:
                value_label = f'Unseen value label: {int(node_pair[1])}'
            except IndexError:
                value_label = f'Invalid index for value label: {int(node_pair[1])}'
            
            print(f'Node pair: ({domain_label}, {value_label})')
            count += 1


if __name__ == "__main__":
    fire.Fire(main)