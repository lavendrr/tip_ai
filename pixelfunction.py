import numpy as np
import read_image
img = read_image.image

"""def pixels:
    for i in img:
        for j in i:
                if i <= 100:
                    p += i
                else:
                    continue


def objects:
    for i in range(p):
        if i != range(p)[-1]:
            if p[i] == p[i+1]
                #...
            else:
                if p[i] == p[0]:
                    #...
                    """

def pixels():
    p = []
    for y in range(np.shape(img)[1]):
        for x in range(np.shape(img)[0]):
            if img[y][x] == 0:
                p.append((x,y))
    return p

print(pixels())
