# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 22:04:12 2017

@author: FLY

extract image patches from HDF5 file for image prediction

-input: HDF5 file including image information
-output: HDF5 files including image patches based on each pixel in image
 
"""

import numpy as np
import h5py
import os


def extInferPatch(hdf5Path,sPatchSize=60,stride=1):
    path=hdf5Path
    newDir=path+"/inference"
    if not os.path.exists(newDir):
        os.makedirs(newDir)
    
    xoffset=sPatchSize/2
    yoffset=sPatchSize/2
    
    patch_size_x=sPatchSize
    patch_size_y=sPatchSize
    
    interval=1000              # 1000 image patches are saved as one HDF5 file
    
    itertest=0
    
    for root,dirs,files in os.walk(path):
        for name in files:
            if name.endswith('.h5'):
                filename=root+"/"+name
                f=h5py.File(filename,'r')
                im_=list(f['data'])[0]
                if len(f)<2:
                    continue
                im_pad=np.pad(im_,((0,0),(xoffset,xoffset-1),(yoffset,yoffset-1)),'reflect')
                img_size=np.shape(im_)[1:]
                iter0=0
                imgPatch=[]
                #print('-----------------------------------')
                #print(img_size[0])
                #print(img_size[1])
                #print('===================================')
                
                for iter1 in range(0,img_size[0],stride):
                    for iter2 in range(0,img_size[1],stride):
                        if len(imgPatch)%interval==0 and len(imgPatch)!=0:
                            h5file=newDir+"/"+name[:-3] + "_"+str(iter0) + ".h5"
                            with h5py.File(h5file,'w') as f:
                                f['data']=imgPatch
                            iter0=iter0+1
                            imgPatch=[]
                            
                            imgPatchTmp=im_pad[:,iter1:iter1+patch_size_x,iter2:iter2+patch_size_y]
                            imgPatch.append(imgPatchTmp)
                            itertest=itertest+1

                        else:
                            imgPatchTmp=im_pad[:,iter1:iter1+patch_size_x,iter2:iter2+patch_size_y]
                            imgPatch.append(imgPatchTmp)
                            itertest=itertest+1
                if len(imgPatch)>0:
                    h5file=newDir+"/"+name[:-3] + "_"+str(iter0) + ".h5"
                    with h5py.File(h5file,'w') as f:
                        f['data']=imgPatch
    
    #print('===============itertest================')
    #print(itertest)
    #print('=======================================')

                        
if __name__ == "__main__":
    path='/disk2/Faliu/cnn/data/infer'           #path to data
    sPatchSize=32
    stride=1
    extInferPatch(path,sPatchSize,stride)
                            
