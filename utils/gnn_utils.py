import torch
import torch.nn.functional as F

def train(model, optimizer, train_data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    pos_edge_scores = model.decoder(z, train_data.pos_edge_label_index)
    neg_edge_scores = model.decoder(z, train_data.neg_edge_label_index)
    pos_probs = torch.sigmoid(pos_edge_scores)
    neg_probs = torch.sigmoid(neg_edge_scores)
    loss = F.binary_cross_entropy(torch.cat([pos_probs, neg_probs]), torch.cat([torch.ones_like(pos_probs), torch.zeros_like(neg_probs)]))
    loss += (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


def create_optimizer(model, optimizer_name='adam', **kwargs):
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs.get('lr', 0.01))
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=kwargs.get('lr', 0.01))
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=kwargs.get('lr', 0.01))
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=kwargs.get('lr', 0.01))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return optimizer


def predict(model, node_features, edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(node_features, edge_index)
        num_nodes = z.size(0)
        pred_edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
        edge_scores = model.decoder(z, pred_edge_index)
        edge_probs = torch.sigmoid(edge_scores)
        return edge_probs