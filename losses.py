import torch
import torch.nn as nn
import torch.nn.functional as F
import math
loss = nn.MSELoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
output = loss(input, target)
l_depth = torch.mean(torch.abs(target - input), axis=-1)
print(l_depth)
output.backward()
print(output)


