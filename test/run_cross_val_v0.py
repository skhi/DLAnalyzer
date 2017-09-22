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
    
    test_files= [] 
    test_files_number = []
    for fold,test in myselector.get_test_dataset():
       hdf5 = HDF5(range(0,8), test_selector_cross_val_config.work_area[fold])	
       for key, val in test.iteritems():
          myDataLoader = DataLoader(key, val)
          bla = myDataLoader.get_data()
          raw, label = ImageAugmenter(bla, datasetImages=val, augType=[ 'default'], rotationRange=30, shiftRange=0.2, batchSize=1).apply_augment()
          hdf5.create(raw, label, key, file_type='test')
       test_file, test_file_number  = hdf5.combine(key='test')   
       test_files.append(test_file)
       test_files_number.append(test_file_number)

    train_files=[] 
    train_files_number=[]
    for fold,train in myselector.get_train_dataset():
       hdf5 = HDF5(range(0,8), test_selector_cross_val_config.work_area[fold])
       for key, val in train.iteritems():
          myDataLoader = DataLoader(key, val)
          bla = myDataLoader.get_data()
  #        raw, label = ImageAugmenter(bla, datasetImages=val, augType=['default', 'rotation','shift', 'flip'], rotationRange=30, shiftRange=0.2, batchSize=1).apply_augment()
          raw, label = ImageAugmenter(bla, datasetImages=val, augType=['default'], rotationRange=30, shiftRange=0.2, batchSize=1).apply_augment() 
          hdf5.create(raw, label, key, file_type='train')
       train_file, train_file_number  = hdf5.combine(key='train')
       train_files.append(train_file)
       train_files_number.append(train_file_number)


    for fold in range(len(train_files)):
#        if fold > 1:
 #         break 
        log.info("train file number: "+str(train_files_number[fold]))
        log.info("test file number: "+str(test_files_number[fold]))
        
        
        myCaffe = CaffeRunner_Cross_Val( test_caffe_cross_val_config._replace(test_dataset=test_files[fold], train_dataset=train_files[fold] ), fold )
        myCaffe.prepare_config()
#        myCaffe.train()
	myCaffe.re_train()
        myCaffe.test(test_files_number[fold], use_trained_weights=False)






