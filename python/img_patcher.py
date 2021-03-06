# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 22:04:12 2017

@author: FLY

extract image patches from HDF5 files for CNN training

input: HDF5 files
output: HDF5 files saving image patches and labels
"""

import re
import numpy as np
import h5py
import glob
import os

class ImgPatcher:
    
    def __init__(self, input, patchSize, patchValue, strideValue, dataType, split="_combined"):

        self.path       = input.split(split)[0]
        self.new_dir    = self.path+"ing"
        self.interval   = patchValue
	self.patch_size = patchSize
	self.stride     = strideValue
        self.input      = input
        self.make_dir_  (self.new_dir)
	self.data_type  = dataType
	print "image patcher was constructed"       

    def patch(self):
        with open(self.input, 'r') as f:
            for line in f:
	        name     = line.split(".h5")[0]+"_.h5"
		line_mod = line.replace("\n", "")
                os.system("cp "+line_mod+" "+name+" ")
                self.extTrainPatch_(name, sPatchSize=self.patch_size, stride=self.stride)

    def make_dir_(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def extTrainPatch_(self, file, sPatchSize,stride):
        
        xoffset=sPatchSize/2
        yoffset=sPatchSize/2
	
	print "file to be opened ", file
        f=h5py.File(file,'r')
        im_=list(f['data'])[0]
        
        labTmp=list(f['label'])[0]
#        if np.isscalar(labTmp):
#            continue
        
        lab_=list(f['label'])[0][0,:,:]
        img_size=np.shape(im_)[1:]
        trainPatch=[]
        trainLabel=[]

	testPatch=[]
        testLabel=[]
       
	if "train" in self.data_type: 
	    row=np.array(list(range(0,img_size[0]-2*xoffset,stride)))+xoffset
            col=np.array(list(range(0,img_size[1]-2*yoffset,stride)))+yoffset
            iter0=0
            for iter1 in list(row):
                for iter2 in list(col):  
                    if (len(trainPatch))%self.interval==0 and len(trainPatch)!=0:
                        h5file= (self.new_dir+os.sep+file.split("/")[-1])[:-4] + "_"+str(iter0) + ".h5"
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
                h5file= (self.new_dir+os.sep+file.split("/")[-1])[:-4] + "_" + str(iter0) + ".h5"
                with h5py.File(h5file,'w') as f:
                     f['data']=trainPatch
                     f['label']=trainLabel

	elif "test" in self.data_type:
	    im_pad=np.pad(im_,((0,0),(xoffset,xoffset-1),(yoffset,yoffset-1)),'reflect')
            img_size=np.shape(im_)[1:]
            iter0=0                
            for iter1 in range(0,img_size[0],stride):
                for iter2 in range(0,img_size[1],stride):
                    if len(testPatch)%self.interval==0 and len(testPatch)!=0:
			h5file= (self.new_dir+os.sep+file.split("/")[-1])[:-4] + "_"+str(iter0) + ".h5"
                        with h5py.File(h5file,'w') as f:
                            f['data']=testPatch
			    f['label']=testLabel
                        iter0=iter0+1
                        testPatch=[]
			testLabel=[]
                            
                        imgPatchTmp=im_pad[:,iter1:iter1+self.patch_size,iter2:iter2+self.patch_size]
			labelTmp=lab_[iter1,iter2]
                        testPatch.append(imgPatchTmp)
			testLabel.append(labelTmp)

                    else:
                        imgPatchTmp=im_pad[:,iter1:iter1+self.patch_size,iter2:iter2+self.patch_size]
			labelTmp=lab_[iter1,iter2]
                        testPatch.append(imgPatchTmp)
                        testLabel.append(labelTmp)
            if len(testPatch)>0:
		h5file= (self.new_dir+os.sep+file.split("/")[-1])[:-4] + "_" + str(iter0) + ".h5"
                with h5py.File(h5file,'w') as f:
                     f['data']=testPatch
		     f['label']=testLabel 
	else:
	   log.info("Wrong data type: Use 'train' or 'test'")
	   os.exit(0)


    def combine(self, key, file_name="_combined.txt"):
        current_dir= os.getcwd()
        os.chdir(self.new_dir)
            
        global files
        with open (self.new_dir+os.sep+key+file_name, 'w') as f:
	        files = [self.new_dir + os.sep + e for e in glob.glob(key+"*h5")]
		files = [e.split('.h5')[0] for e in files]
		files.sort(key=natural_keys)
		files = [e+str('.h5') for e in files]		
                f.write( '\n'.join( files  )  )
        path_to_file = self.new_dir+os.sep+key+file_name
            
	os.chdir(current_dir)
        return ( path_to_file, len(files))



## list sorting related 


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('_', text) ]
