import numpy as np
import pdb

a = np.arange(12).reshape([2,6])
print(a)
b = np.transpose(np.reshape(a,(-1,3,2)),(0,2,1))
print(b)
pdb.set_trace()
