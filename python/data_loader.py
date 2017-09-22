#!/usr/bin/env python

import os
from os import chdir, listdir
import glob
from skimage import data, color, img_as_float, io
import numpy as np
import csv
import cv2
from PIL import Image
from logger import log


folder = ['Images', 'Mask']
#folder = ['images', 'mask']

class DataLoader:
    
    ''' this is the DataLoader class '''
    
    def __init__(self, datasetPath, datasetImages,  imgHeight=360, imgWidth=480):
        
        self.np_image       = np.array([], dtype = np.float32)
        self.deepview       = ''
        self.np_mask        = np.array([], dtype = np.float32)
        self.height         = imgHeight
        self.width          = imgWidth
        self.path           = datasetPath
        self.images         = datasetImages
        
        def load_data(files):
            selected_files = []
            for file in files:
                picFound = False
                for selected_file in glob.glob(file+"*"):
                    selected_files.append(selected_file)
                    picFound = True
                # An error message if the picture indicated in the pics.txt file wasn't found in the directory.
                if picFound == False:
                    log.error(str(file)+" file wasn't found!")
            return selected_files
               
        # change the directory to where the dataset is
        current_dir = os.getcwd()
        chdir(self.path)
        
        # move to the "image" directory first
        chdir(folder[0])

        # load the raw images
        self.np_image = [np.array(Image.open(i), dtype = np.float32) for i in load_data(filter(lambda s: not s.startswith('DeepviewOutput'), self.images))]


        # load the DeepView image
        # data.load function needs the path to the picture, otherwise it always looks at skimage/data/ directory by default
        self.np_deepview = data.load( os.getcwd()+os.sep+load_data(filter(lambda s: s.startswith('DeepviewOutput'), self.images))[0] )
        # now move to the "mask" directory

        # load the mask
        # data.load function needs the path to the picture, otherwise it always looks at skimage/data/ directory by default
        chdir('../'+folder[1])
        self.np_mask =  data.load(os.getcwd()+os.sep+load_data( ['mask'])[0])
        chdir(current_dir)
    
    
    def get_data(self):

        #### collapse trailing dimensions
        self.np_image = np.squeeze(self.np_image) # dim = np_image.size * H * W
        
        #### load DeepView output
        self.np_deepview = color.rgb2gray(self.np_deepview) # convert DeepView output to 1 channel gray image
        self.np_deepview = np.reshape(self.np_deepview, (1,self.np_deepview.shape[0],self.np_deepview.shape[1]) )
        
        #### merge 8 MSI and DeepView into 9 channel images

        
        self.np_image = np.append(self.np_image,self.np_deepview,axis=0) # dim(F) = 9 * H * W

        #### swap axes
        self.np_image = np.swapaxes(self.np_image, 0, 2)
        self.np_image = np.swapaxes(self.np_image, 0, 1)  # dim(F)= H * W * 9

        ### now mask has 4 dimensions, remove the last dimension
        self.np_mask = self.np_mask[..., np.newaxis]

        #### merge mask into F, total 12 channels
        self.np_image = np.append(self.np_image, self.np_mask, axis=2)   # dim(F) = H * W * 12

        Z = np.empty( (self.height, self.width, self.np_image.shape[2]) )
        
        ### resize to H * W * 12
        for i in range(self.np_image.shape[2]):
            Z[:,:,i] = cv2.resize(self.np_image[:,:,i], (self.width, self.height), interpolation = cv2.INTER_NEAREST)
                #print "Z size, ndim {}:{}".format(Z.size, Z.ndim)
                #print Z[:,:,0:]
        return Z

