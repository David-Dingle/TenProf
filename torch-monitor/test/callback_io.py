##
# A simple test case for examining the functionality of callback inputs/outputs capturing
# The code was from: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
#
# #

import sys
import torch
import torch.nn as nn
device = str(sys.argv[1])
device = torch.device(device)

x = torch.ones(10, 5, requires_grad=False, device=device)
w = torch.ones(1, 5, requires_grad=False, device=device)
bias = torch.ones(10, 1, requires_grad=False, device=device)

a = torch.ones(10, 5, requires_grad=False, device=device)
b = torch.ones(1, 5, requires_grad=False, device=device)

c = torch.ones(10, 5, requires_grad=False, device=device)
d = torch.ones(1, 5, requires_grad=False, device=device)

e = a.T
f = b.T

g = e[1 : -1, : ]
h = e[ : ,1 : -1]

out = torch._C._nn.linear(input=x, weight=w, bias=bias)
