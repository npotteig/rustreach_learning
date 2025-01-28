import torch
import argparse

from mfnlc.evaluation.model import load_model
from mfnlc.learn.lyapunov_td3 import LyapunovTD3
from mfnlc.config import get_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALGO = "lyapunov_td3"
ROBOT_NAME = "Quadcopter"

if __name__ == "__main__":
    # get command args
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", type=str, default=ROBOT_NAME)
    args = parser.parse_args()
    
    env_name = f"{args.robot_name}-no-obst"
    
    if args.robot_name == "Quadcopter":
        state_size = 12
    elif args.robot_name == "Bicycle":
        state_size = 4
    
    model: LyapunovTD3 = load_model(env_name, ALGO)
    dummy_input = torch.randn(1, state_size).to(device)
    torch.onnx.export(
        model.policy.actor,               # The model to export
        dummy_input,         # An example input tensor
        get_path(args.robot_name, ALGO, "model_actor"),        # The file where the ONNX model will be saved
        export_params=True,  # Store the trained parameters (weights) inside the ONNX model
        opset_version=16,    # The ONNX version to export to (11 is a good default)
    )
    