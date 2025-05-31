
import hashlib
import torch
from models.federated_model import RiskPredictionMLP

def hash_model_weights(model_path):
    model = RiskPredictionMLP(input_dim=4)
    model.load_state_dict(torch.load(model_path))
    all_params = []
    for param_tensor in model.state_dict().values():
        all_params.append(param_tensor.cpu().numpy().tobytes())
    concatenated = b''.join(all_params)
    return hashlib.sha256(concatenated).hexdigest()

def simulate_zk_proof(model_path):
    model_hash = hash_model_weights(model_path)
    proof = f"ZK-PROOF-SIMULATED::{model_path}::SHA256::{model_hash}"
    print("Simulated Zero-Knowledge Proof:")
    print(proof)
    return proof

if __name__ == "__main__":
    simulate_zk_proof('global_model.pt')
