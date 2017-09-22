import h5py
import os
from os import chdir, listdir
import numpy as np
import sys
import subprocess
import glob
from logger import log
from mean_sub import MeanSub


class HDF5:
    
    def __init__(self,  numChannels,  baseLocation, fileLocation="HDF5Files"):
    
        self.baseloc     = baseLocation
        self.saveloc     = fileLocation
        
        if type(numChannels) is list:
            self.channel = numChannels #range(0,8)#[5,1,0]#
        else:
            logging.error('The variable \'numChannels\' must be a range from a minimum of 0 to a maximum of 9 that consists of\n the nuber of features you would like to include. For all MSI images use range(0,8)')
            sys.exit(0)
        
        self.__create_workarea()

    def __create_workarea(self):
        dir_name = self.baseloc + os.sep + self.saveloc
        if not os.path.isdir(dir_name):
            log.info("Creating "+self.saveloc+" area")
            os.mkdir(dir_name)
        else:
            log.error(self.saveloc+" already there")

    def create(self, raw, label, file, file_type):
        
        file  = file.rstrip('/').split('/')
        str_list = list()
        for piece  in file:
            str_list.append(piece)
        h5name='_'.join(str_list)
        
        # Reshape data
        data = np.transpose(raw,(3,2,0,1))   # data shape needs to be 15 * 9 * H * W
        Y    = np.transpose(label,(3,2,0,1))  # data label (mask)
            
        # Decide which channels (feautres) to use
        data = data[:,self.channel,:,:] # new data shape
            
        # If you do not have enough memory split data into
        # multiple batches and generate multiple separate h5 files
        with h5py.File( self.baseloc + os.sep + self.saveloc + os.sep + file_type + str('_') + h5name.lstrip('/')+str('.h5'), 'w') as H:
            # data and label name matches "top" in prototxt.txt in the data layer
            H.create_dataset( 'data', data = data.astype(np.float) ) # note the name data given to the dataset!
            H.create_dataset( 'label', data = Y.astype(np.float)) # note the name label given to the dataset!
        
        log.info( "Each image in the hdf5 database has dimensions: "+ str(Y[0].shape))
        log.info( "Each set of images from one image capture in the hdf5 database has dimensions: " + str( data[0].shape) )
        log.info( "In the hdf5 database there are a total of " + str(data.shape[0]) + " images." )
        log.info( "The mask has these unique classes: " + str( np.unique(label) ) )
        log.info( "The hdf5 database in located in: " + str(self.saveloc) )
        log.info( "self.saveloc " + str(self.saveloc) )
        log.info( "h5name " + str(h5name) )
    
    def apply_mean_sub(self, file):
        current_dir= os.getcwd()
        hdf5_dir   = self.baseloc + os.sep + self.saveloc
        MeanSub(hdf5_dir, file).mean_sub()
    

    def combine(self, key, file_name="_combined.txt"):
        current_dir= os.getcwd()
        hdf5_dir   = self.baseloc + os.sep + self.saveloc
        chdir(hdf5_dir)
        
        sub_name = ''
	if '*' in key:
 	    sub_name = key.split('*')[0]	 
	else:
	    sub_name = key  
        
	global files 
        with open (sub_name+file_name, 'w') as f:
	    files = [os.getcwd() + os.sep + e for e in glob.glob(key+"*h5")]
            f.write( '\n'.join( files  )  )
        path_to_file = current_dir +os.sep + hdf5_dir+os.sep+sub_name+file_name
        chdir(current_dir)
        
        return (path_to_file, len(files)) 



    def kill(self):
	del self


