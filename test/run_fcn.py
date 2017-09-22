#!/usr/bin/env python
import os, sys
import logging
from selector_config import test_selector_config
from caffe_config import test_caffe_config_fcn
from selector import Selector
from data_loader import DataLoader
from data_augmenter import ImageAugmenter
from caffe_runner import CaffeRunner
from hdf5_creator import HDF5
from img_patcher import ImgPatcher
from logger import log
import gc


if __name__ == "__main__":
    

    myselector = Selector(test_selector_config)
    
    
    test = myselector.get_test_dataset()
    train = myselector.get_train_dataset()
    
    hdf5 = HDF5(range(0,8), test_selector_config.work_area)
    
    for key, values in test.iteritems():
        myDataLoader = DataLoader(key, values)
        bla = myDataLoader.get_data()
        raw, label = ImageAugmenter(bla, datasetImages=values, augType=[ 'rotation','shift', 'flip'], rotationRange=30, shiftRange=0.2, batchSize=1).apply_augment() # , 'shift', 'flip'
#	raw, label = ImageAugmenter(bla, datasetImages=values, augType=[ 'default'], rotationRange=30, shiftRange=0.2, batchSize=1).apply_augment() # , 'shift', 'flip'
        hdf5.create(raw, label, key, file_type='test')

    
    
    for key, values in train.iteritems():
        myDataLoader = DataLoader(key, values)
        bla = myDataLoader.get_data()
        raw, label = ImageAugmenter(bla, datasetImages=values, augType=[ 'rotation','shift', 'flip'], rotationRange=30, shiftRange=0.2, batchSize=1).apply_augment() # , 'shift', 'flip'
        hdf5.create(raw, label, key, file_type='train')

    name = '/disk2/nik/Analyzer/test/file_means.txt'
#    hdf5.apply_mean_sub(name)

#    test_file = hdf5.combine(key='test*mean')
#    train_file = hdf5.combine(key='train*mean')
  
    test_file, test_file_number  = hdf5.combine(key='test')
    train_file, train_file_number = hdf5.combine(key='train')  

    log.info("train file number: "+str(train_file_number))
    log.info("test file number: "+str(test_file_number))
    test_caffe_config_fcn = test_caffe_config_fcn._replace(test_dataset=train_file, train_dataset=train_file )

    myCaffe = CaffeRunner(test_caffe_config_fcn)
    myCaffe.prepare_config()
#    myCaffe.train()
#    myCaffe.re_train()	

    myCaffe.test(train_file_number*3, use_trained_weights=True)
#    myCaffe.test_cnn( test_file_number , 173, 1000, use_trained_weights=True)
