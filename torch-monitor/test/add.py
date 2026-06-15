import torch
import sys

device = str(sys.argv[1])
device = torch.device(device) 
left = torch.zeros(1000, device=device, requires_grad=True)
right = torch.zeros(1000, device=device, requires_grad=True)
grad = torch.zeros(1000, device=device)

for _ in range(30):
    output = torch.add(left, right)
    output.backward(grad)
