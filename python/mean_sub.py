# -*- coding: utf-8 -*-
"""
    Created on Wed Apr  5 11:59:35 2017
    
    @author: Faliu Yi
    subtract the mean value for each channel of images in HDF5
    
    """
import numpy as np
import os
import h5py
import cv2
from logger import log


class MeanSub:

    def __init__(self, hdf5Dir, avgValFile):

        self.dir        = hdf5Dir
        self.avg_val    = self.extract_values_(avgValFile)

    # open file and extract the value and then append to list
    def extract_values_(self, file):
        list=[]
        with open(file,'r') as f:
           for line in f:
              list.append(float(line.split(" ")[1]))
        return list
                            
    
                            
    def mean_sub(self):
        """
            input:
            -pathtoHDF5: the path to the HDF5 files
            -newDir: the path to a new folder that stores the new HDF5 files with
            mean value subtracted. A new folder is created if it is not existed
            
            """
        if not os.path.exists(self.dir):
            log.info("Directory doesn't exist, exiting!!!")
            os.exit()
        
        for root,dirs,files in os.walk(self.dir):
            for name in files:
                if name.endswith('.h5'):
                    filename=root+"/"+name
                    f=h5py.File(filename,'r')
                    im_=list(f['data'])
                    la_=list(f['label'])
                    data=[]
                    label=[]
                    for iter in range(len(im_)):
                        im_tmp=im_[iter]
                        la_tmp=la_[iter]
                        mSize=im_tmp.shape
                        data_temp=np.zeros((mSize))
                        # 8 channels
                        for iter2 in range(len(im_tmp)):
                            temp=im_tmp[iter2]-self.avg_val[iter2]
                            data_temp[iter2,:,:]=temp
                        data.append(data_temp)
                        kernel=np.ones((7,7),np.uint8)
                        imgClosing=cv2.morphologyEx(la_tmp,cv2.MORPH_CLOSE,kernel)
                        label.append(imgClosing)
                        h5_fn=filename.split(".h5")[0]+str("_meansub.h5")
                        with h5py.File(h5_fn,'w') as f:
                            f['data']=data
                            f['label']=label

