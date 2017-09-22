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
    
    train_files=[] 
    train_files_number=[]
    for fold,train in myselector.get_train_dataset():
       hdf5 = HDF5(range(0,8), test_selector_cross_val_config.work_area[fold])
       for key, val in train.iteritems():
          myDataLoader = DataLoader(key, val)
          bla = myDataLoader.get_data()
          raw, label = ImageAugmenter(bla, datasetImages=val, augType=[ 'rotation','shift', 'flip'], rotationRange=30, shiftRange=0.2, batchSize=1).apply_augment()
          hdf5.create(raw, label, key, file_type='train')
       train_file, train_file_number  = hdf5.combine(key='train')
       train_files.append(train_file)
       train_files_number.append(train_file_number)


    test_files='/disk2/nik/Analyzer/test/sample_size_test/test_combined.txt'
    test_files_number = 4

    for fold in range(len(train_files)):
        
        log.info("train file number: "+str(train_files_number[fold]))
        log.info("test file number: "+str(test_files_number))
        
        
        myCaffe = CaffeRunner_Cross_Val( test_caffe_cross_val_config._replace(test_dataset=test_files, train_dataset=train_files[fold] ), fold )
        myCaffe.prepare_config()
#        myCaffe.train()
	myCaffe.re_train()
        myCaffe.test(test_files_number, use_trained_weights=False)






