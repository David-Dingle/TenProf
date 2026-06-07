import torch
import sys

device = str(sys.argv[1])
device = torch.device(device) 
left = torch.zeros(100, device=device, requires_grad=True)
right = torch.zeros(100, device=device, requires_grad=True)
grad = torch.zeros(100, device=device)

for _ in range(3):
    output = torch.add(left, right)
    output.backward(grad)
