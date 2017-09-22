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

# for threading
import Queue
import threading

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
          raw, label = ImageAugmenter(bla, datasetImages=val, augType=['default'], rotationRange=30, shiftRange=0.2, batchSize=1).apply_augment() 
          hdf5.create(raw, label, key, file_type='train')
       train_file, train_file_number  = hdf5.combine(key='train')
       train_files.append(train_file)
       train_files_number.append(train_file_number)

instances = [CaffeRunner_Cross_Val( test_caffe_cross_val_config._replace(test_dataset=test_files[fold], train_dataset=train_files[fold] ), fold ) for fold in range(len(train_files))]

q = Queue.Queue()

def submit_jobs(q,th, gpu):
    q.put(th.re_train(gpu))

for n, t in enumerate(instances):
    log.info("train file number: "+str(train_files_number[n]))
    log.info("test file number: "+str(test_files_number[n]))
    gpu = 0
    if n%2 == 0:
        gpu = 0
    else:
        gpu = 1
    t.prepare_config()
    t = threading.Thread(target=submit_jobs, args = (q,t,gpu))
    t.daemon = True
    t.start()

    print "-----------------"
    if n > 1:
        break
s = q.get()
#    t = threading.Thread(target=th.re_train())
#    t.daemon = True
#    t.start()


#    for fold in range(len(train_files)):
#        log.info("train file number: "+str(train_files_number[fold]))
#        log.info("test file number: "+str(test_files_number[fold]))
        
        
#        myCaffe = CaffeRunner_Cross_Val( test_caffe_cross_val_config._replace(test_dataset=test_files[fold], train_dataset=train_files[fold] ), fold )
#        myCaffe.prepare_config()
#	myCaffe.re_train()
#        myCaffe.test(test_files_number[fold], use_trained_weights=False)






