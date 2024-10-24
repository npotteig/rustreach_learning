import torch

from mfnlc.evaluation.model import load_model
from mfnlc.learn.lyapunov_td3 import LyapunovTD3
from mfnlc.config import get_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENV_NAME = "Quadcopter-no-obst"
ALGO = "lyapunov_td3"
ROBOT_NAME = "Quadcopter"

STATE_SIZE = 12

if __name__ == "__main__":
    model: LyapunovTD3 = load_model(ENV_NAME, ALGO)
    dummy_input = torch.randn(1, STATE_SIZE).to(device)
    torch.onnx.export(
        model.policy.actor,               # The model to export
        dummy_input,         # An example input tensor
        get_path(ROBOT_NAME, ALGO, "model_actor"),        # The file where the ONNX model will be saved
        export_params=True,  # Store the trained parameters (weights) inside the ONNX model
        opset_version=16,    # The ONNX version to export to (11 is a good default)
    )
    