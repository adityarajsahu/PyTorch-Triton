import torch
from torch2trt import torch2trt

from utils.model import ImageClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ImageClassifier()
model.to(device)
model.load_state_dict(torch.load("checkpoints/model.pt"))
model.eval()

x = torch.randn(1, 3, 32, 32)
x = x.to(device)
model_trt = torch2trt(model, [x])
# print(model)

with open("deploy_triton/model.plan", "wb") as f:
    f.write(model_trt.engine.serialize())