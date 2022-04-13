# main.py

import torch

# this should print "True" if the environment is configured correct with GPU access
print(torch.cuda.is_available())
