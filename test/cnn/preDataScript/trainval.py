# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:20:24 2017

@author: smd
"""

import os

def trainValPath(hdf5path):
    path=hdf5path
    f1=path + "/train.txt"
    f2=path + "/test.txt"
    ftrain=open(f1,"w")
    ftest=open(f2,"w")
    for root,dirs,files in os.walk(path):
        for name in files:
            if name.endswith('.h5'):
                fileNamePath=root + "/" + name
                if "train" in name:
                    ftrain.write("%s" % fileNamePath)
                    ftrain.write('\n')
                else:
                    ftest.write("%s" % fileNamePath)
                    ftest.write('\n')
    ftrain.close
    ftest.close
    
if __name__ == "__main__":
    path='/disk2/Faliu/cnn/data/trainval/training'          #path to data
    trainValPath(path)
    





    
        
            
            
            
                
                
            
                
                


                
                
                
