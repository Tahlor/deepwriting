import os
from shutil import copyfile
import multiprocessing

path = r"/media/data/GitHub/deepwriting/data/IAM_online/lineImages-all/lineImages"
output = r"/media/SuperComputerCompute/research/handwriting/MUNIT/datasets/handwriting/trainC"
i=1
#os.mkdir(output)
for d,s,fs in os.walk(path):
    for f in fs:
        if f[-4:]==".tif":
            i+=1
            if i % 100 == 0:
                print(i)
            old = os.path.join(d,f)
            new = os.path.join(output,f)
            copyfile(old, new)
