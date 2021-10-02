import os
import shutil
import torch

#for line in os.listdir("meta/files"):
#    if(not os.path.isdir("meta/speakers/" + line[3:6])):
#        os.makedirs("meta/speakers/" + line[3:6])
#    shutil.copy("meta/files/"+line,"meta/speakers/" + line[3:6])

input = torch.randn(3, 5, requires_grad=True)
target = torch.randint(5, (3,), dtype=torch.int64)
loss.backward()