#!/usr/bin/env python
import os, sys
import logging
from selector_cross_val_config import test_selector_cross_val_config
from caffe_cross_val_config import test_caffe_cross_val_config
from selector_cross_val import Selector_Cross_Val
from data_loader import DataLoader
from data_augmenter import ImageAugmenter
from caffe_cross_val_runner import CaffeRunner_Cross_Val
from hdf5_creator import HDF5
from img_patcher import ImgPatcher
from logger import log
import gc


if __name__ == "__main__":
    

    myselector = Selector_Cross_Val(test_selector_cross_val_config)
    
    test_files='/disk2/nik/Analyzer/test/sample_size_test/test_combined.txt'
    test_files_number = 4


    train_files = ['/disk2/nik/Analyzer/test/HDF5Files/train_combined_15_2.txt']
#'/disk2/nik/Analyzer/test/HDF5Files/samples_25/train_combined.txt']
#'/disk2/nik/Analyzer/test/HDF5Files/train_combined_25.txt', 
#                   '/disk2/nik/Analyzer/test/HDF5Files/samples_50/train_combined.txt', 
#                   '/disk2/nik/Analyzer/test/HDF5Files/samples_100/train_combined.txt']

    train_files_number = [15]

    for fold in range(len(train_files)):
        
        log.info("train file number: "+str(train_files_number[fold]))
        log.info("test file number: "+str(test_files_number))
        
        
        myCaffe = CaffeRunner_Cross_Val( test_caffe_cross_val_config._replace(test_dataset=test_files, train_dataset=train_files[fold] ), fold )
        myCaffe.prepare_config()
#        myCaffe.train()
	myCaffe.re_train()
        myCaffe.test(test_files_number, use_trained_weights=False)






