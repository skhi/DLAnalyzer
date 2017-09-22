# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 22:04:12 2017

@author: FLY

extract image patches from HDF5 files for CNN training

input: HDF5 files
output: HDF5 files saving image patches and labels
"""

import numpy as np
import h5py
import os

def extTrainPatch(hdf5Path,sPatchSize=60,stride=1):
    path=hdf5Path
    newDir=path+"/training"
    if not os.path.exists(newDir):
        os.makedirs(newDir)
    
    xoffset=sPatchSize/2
    yoffset=sPatchSize/2
    
    interval=1000              # 1000 image patches are saved as one HDF5 file
    
    
    for root,dirs,files in os.walk(path):
        for name in files:
            if name.endswith('.h5'):
                filename=root+"/"+name
                f=h5py.File(filename,'r')
                im_=list(f['data'])[0]
                
                labTmp=list(f['label'])[0]
                if np.isscalar(labTmp):
                    continue
                
                lab_=list(f['label'])[0][0,:,:]
                img_size=np.shape(im_)[1:]
                trainPatch=[]
                trainLabel=[]
                row=np.array(list(range(0,img_size[0]-2*xoffset,stride)))+xoffset
                col=np.array(list(range(0,img_size[1]-2*yoffset,stride)))+yoffset
                iter0=0
                for iter1 in list(row):
                    for iter2 in list(col):  
                        if (len(trainPatch))%interval==0 and len(trainPatch)!=0:
                            h5file=newDir+"/"+name[:-3] + "_"+str(iter0) + ".h5"
                            with h5py.File(h5file,'w') as f:
                                f['data']=trainPatch
                                f['label']=trainLabel
                            iter0=iter0+1
                            trainPatch=[]
                            trainLabel=[]
                        else:
                            dataTmp=im_[:,iter1-xoffset:iter1+xoffset,iter2-yoffset:iter2+yoffset]
                            labelTmp=lab_[iter1,iter2]
                            trainPatch.append(dataTmp)
                            trainLabel.append(labelTmp)
                
                if (len(trainPatch))>0:
                    h5file=newDir+"/"+name[:-3] + "_" + str(iter0) + ".h5"
                    with h5py.File(h5file,'w') as f:
                        f['data']=trainPatch
                        f['label']=trainLabel
                        
if __name__ == "__main__":
    path='/disk2/Faliu/cnn/data/trainval'           #path to data
    sPatchSize=32
    stride=5
    extTrainPatch(path,sPatchSize,stride)
