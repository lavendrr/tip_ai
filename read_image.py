# Lloyd Cloer
# code adapted from: https://stackoverflow.com/questions/32946436/read-img-medical-image-without-header-in-python

import numpy as np
import matplotlib.pyplot as plt

input_filename = "comb.img"
shape = (512, 512)
d_type = np.dtype(np.uint8)

with open(input_filename, 'r') as f:
    data = np.fromfile(f, dtype = d_type)
a = data[512:]
image = a.reshape(shape)
print(image)


plt.imshow(image)
plt.show()
