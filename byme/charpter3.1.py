import torch
import torch.nn as nn
import numpy as np

data=np.loadtxt("german.data-numeric")

n,l=data.shape
for j in range(l-1):
    meanVal=np.mean(data[:,j])
    stdVal=np.std(data[:,j])
    data[:,j]=(data[:,j]-meanVal)/stdVal