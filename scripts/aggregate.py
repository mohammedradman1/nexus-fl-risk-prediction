
import torch
import os
from models.federated_model import RiskPredictionMLP

def load_weights(path_list):
    weights = []
    for path in path_list:
        model = RiskPredictionMLP(input_dim=4)
        model.load_state_dict(torch.load(path))
        weights.append(model.state_dict())
    return weights

def average_weights(weights_list):
    avg_weights = {}
    for key in weights_list[0].keys():
        avg_weights[key] = sum([weights[key] for weights in weights_list]) / len(weights_list)
    return avg_weights

def save_aggregated_model(avg_weights, save_path='global_model.pt'):
    global_model = RiskPredictionMLP(input_dim=4)
    global_model.load_state_dict(avg_weights)
    torch.save(global_model.state_dict(), save_path)
    print(f"Global model saved to {save_path}")

if __name__ == "__main__":
    # Example usage with one client (extendable to more)
    client_weight_files = ['client_weights.pt']  # Replace or extend with more clients
    weights = load_weights(client_weight_files)
    avg_weights = average_weights(weights)
    save_aggregated_model(avg_weights)
