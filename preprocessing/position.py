# -*- coding: utf-8 -*-
import os
f=open("./test5.csv", "a")
dirpath = "/home/operations/Documents/20230712/test"
for root,dirs,files in os.walk(dirpath):
    for file in files:
        f.writelines(os.path.join(root,file)+"\n")
