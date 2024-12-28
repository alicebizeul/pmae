from scipy.fftpack import dct, idct
import torch
from torchvision import datasets, transforms
import random
from sklearn.metrics import mean_squared_error
import numpy as np
import os 
from PIL import Image
import glob 
import matplotlib.pyplot as plt
# coefs 

list_coef = glob.glob("/cluster/home/abizeul/mae/tools/dct/coefs/*.npy")


all_coefs = []
all_sum = []
for c in list_coef:
    c = np.load(c)
    # for i in range(3):
    #     c[i] = (c[i] - np.min(c[i]))/(np.max(c[i])-np.min(c[i]))
    #     c[i] = c[i]/np.sum(c[i])
    all_coefs.append(c)
    # print(c.shape)
    # all_sum.append(np.sum(np.sum(c,1),1))

all_coefs = np.stack(all_coefs,0)
print("All coef",all_coefs.shape,np.sum(all_coefs,0).shape,np.sum(all_coefs,0).flatten().shape)
mean_all_coeffs = np.sum(all_coefs,0).flatten()/(np.sum(all_coefs))
order = np.argsort(mean_all_coeffs)
print("order",order.shape)
np.save("/cluster/home/abizeul/mae/tools/dct/ordered_dct.npy",order)

plt.plot(mean_all_coeffs[order])
plt.savefig("/cluster/home/abizeul/mae/tools/dct/ordered_dct.png")
plt.close()

# plt.plot(all_sum)
# plt.savefig("/cluster/home/abizeul/mae/tools/dct/sum.png")
# plt.close()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(mean_all_coeffs[order][10:], color='blue', lw=2)
ax.set_yscale('log')

plt.savefig("/cluster/home/abizeul/mae/tools/dct/ordered_dct_log.png")