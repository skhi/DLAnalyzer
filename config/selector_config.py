#!/usr/bin/env python
from selector_config_template import selector_config

# General comment: Here you can define as many instancies of the "selector_config" as you want. Each instance should be imported and then used separately in the run.py file. In this example, there is only one instance, called "test_selector_config"

test_selector_config = selector_config._replace(work_area='output',
                                         job_type='',
                                         input_csv ='CRF_full_data_corrected_v3.csv', # 'CRF_DU.csv', # 'CRF_full_data_corrected.csv', #'CRF_full_data_sample_size_train.csv', #'CRF_full_data_corrected.csv', #'CRF_data.csv', #'CRF_full_data_corrected.csv', #'CRF_full_data_corrected.csv', #  'CRF_data.csv',# 'CRF_full_data_corrected.csv', #'CRF_full_data.csv', # 'CRF_data.csv',
                                         time_interval=0,
                                         split_train_test=0.70,
                                         split_random_seed = 100,
                                         split_method='patientid', # patientid #'burnarea', #'burnarea',
                                         keep_logs=False,
                                         gpic_selection_file= '/disk2/nik/Analyzer/test/pics.txt' # 'pics.txt' #'/disk2/nik/Analyzer/test/pics.txt'
                                         )






