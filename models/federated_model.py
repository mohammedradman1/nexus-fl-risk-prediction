import torch
import torch.nn as nn

class RiskPredictionMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=1):
        super(RiskPredictionMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, x):
        return self.model(x)
