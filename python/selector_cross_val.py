#!/usr/bin/env python
import os, sys
import collections
from logger import log
import pandas as pd
from sklearn.model_selection import KFold

class Selector_Cross_Val:
    '''Selector class'''
    def __init__(self, namedtupleConfig):
        
        self.df                     = pd.DataFrame()
        self.dataset                = collections.defaultdict(list)
        self.images                 = {}
        self.train_dataset          = []
        self.test_dataset           = []
        self.patient_id             = []
        self.work_area              = namedtupleConfig.work_area
        self.csv_file               = namedtupleConfig.input_csv
        self.time_interval          = namedtupleConfig.time_interval
        self.split_method           = namedtupleConfig.split_method
        self.n_split                = namedtupleConfig.n_split
        self.shuffle                = namedtupleConfig.shuffle
        self.random_state           = namedtupleConfig.random_state
        self.image_global_file      = namedtupleConfig.gpic_selection_file
        self.global_image_list      = self.load_config_images()
        
        self.create_workarea()
        self.load_csv()
        self.load_dataset_and_images()
        self.train_n_kfold = self.n_kfold()
	self.test_n_kfold  = self.n_kfold()
        log.info("The framework runs with this configuration: \n" +
             "\n".join(  "{}: {}".format(k, v) for k, v in namedtupleConfig.__dict__.items()) )
             
    def load_csv(self):
        if os.path.exists(self.csv_file):
            self.df = pd.read_csv(self.csv_file)
        else:
            log.error("The csv file does not exist!")

    def load_dataset_and_images(self):
        for _, row in self.df.iterrows():
            file_name    = row['Datasets']
            pics_name    = row['Pictures']
            initial_time = row['Enrollment_Date'] # this format is essumed: 2017-02-01_09.38.06.849
            
            crf_year  = int( initial_time.split('/')[2] )
            crf_month = int( initial_time.split('/')[0] )
            crf_day   = int( initial_time.split('/')[1] )
            
            key = ''
            with open(file_name, 'r') as d_f, open(pics_name, 'r') as p_f :
                for line in d_f:
                    # this format is essumed: 2017-02-01_09.38.06.849  dataset
                    pic_year  = int( (line.split(' ')[0]).split('-')[0] )
                    pic_month = int( (line.split(' ')[0]).split('-')[1] )
                    pic_day   = int( (line.split(' ')[0]).split('-')[2].split('_')[0] )
                    
                    # dataset is a dictionary: key = time stamps and value is a path to pics
                    # key = line.split(' ')[1].split('/')[1] + os.sep + line.split(' ')[1].split('/')[2] + os.sep + line.split(' ')[1].split('/')[3]
                    str_list = line.split(' ')[1].rstrip('\n').split('Patient')
                    key = "/".join( [ str_list[0], 'Patient'+str_list[1].split('/')[0] ] )
                    self.dataset[key].append( "/".join( str_list[1].split('/')[1:] ))
                    self.patient_id.append( 'Patient'+key.split('Patient')[1].split('/')[0] )
                    # load pictures in the dic as a list
                    self.images[key] = self.global_image_list if len(self.global_image_list) else [line.rstrip('\n') for line in p_f]

        # Make sure IDs are unique in the list
        self.patient_id =  list(set(self.patient_id))
        
        if self.split_method == 'patientid':
            tmp_dic = {}
            for id in self.patient_id:
                for key, value in self.dataset.iteritems():
                    if id in key:
                       tmp_dic[key.split(id)[0]+id] = [key.split(id)[1]+"/"+x for x in value]
            # Assign the tmp dictionary to the original dataset
            self.dataset = tmp_dic


    def create_workarea(self):
        root_path = os.getcwd()
        for folder in self.work_area:
            os.mkdir(os.path.join(root_path,folder))


    def load_config_images(self):
        # Make sure that the config file includes valid pics txt file (this will be used as a global file)
        image_list = []
        if self.image_global_file.strip():
            log.info("Picture selection is done from the global 'gpic_selection_file' file!")
            with open(self.image_global_file, 'rb') as txtfile:
                image_list = [line.rstrip('\n') for line in txtfile]
        else:
            log.info("Global 'gpic_selection_file' file wasn't indicated, pictures from the dataset pics.txt file will be used!")
        return image_list

    def n_kfold(self):
        sample = []
        for key, values in self.dataset.iteritems():
            if self.split_method == 'session':
                for value in values:
                    sample.append(key + os.sep + value)
            elif self.split_method == 'burnarea':
                sample.append(key)
            elif self.split_method == 'patientid':
                sample.append(key)
            else:
                log.error("splitMethod option is wrong!")
    
        kf = KFold(n_splits=self.n_split, shuffle=self.shuffle, random_state=self.random_state)
        for train_index, test_index in kf.split(sample):
                train_set = [sample[x] for x in train_index]
                test_set  = [sample[x] for x in test_index]
                yield train_set,test_set 


    def get_train_dataset(self):
        
        for fold in range(self.n_split): 
            sample = {}
            train_set,_ = self.train_n_kfold.next()
            for key, values in self.dataset.iteritems():
                if self.split_method == 'session':
                    for value in values:
                        newkey = key + os.sep + value
                        if newkey in train_set:
                            sample[newkey] = self.images[key]
                elif self.split_method == 'burnarea':
                    if key in train_set:
                        for value in values:
                            sample[key + os.sep + value] = self.images[key]
                elif self.split_method == 'patientid':
                    if key in self.train_dataset:
                        for value in values:
                            sample[key + os.sep + value] = self.images[key]
                    
            yield fold , sample


    def get_test_dataset(self):
        
        for fold in range(self.n_split):
            sample = {}
	    _,test_set = self.test_n_kfold.next()
            for key, values in self.dataset.iteritems():
                if self.split_method == 'session':
                    for value in values:
                        newkey = key + os.sep + value
                        if newkey in test_set:
                            sample[newkey] = self.images[key]
                elif self.split_method == 'burnarea':
                    if key in test_set:
                        for value in values:
                            sample[key + os.sep + value] = self.images[key]
                elif self.split_method == 'patientid':
                    if key in self.test_dataset:
                        for value in values:
                            sample[key + os.sep + value] = self.images[key]
                                
            yield fold, sample


