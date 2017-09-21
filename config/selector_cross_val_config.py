#!/usr/bin/env python
from selector_config_template import selector_cross_val_config

# General comment: Here you can define as many instancies of the "selector_config" as you want. Each instance should be imported and then used separately in the run.py file. In this example, there is only one instance, called "test_selector_config"

test_selector_cross_val_config = selector_cross_val_config._replace(work_area=['pocwisc1', 'pocwisc2', 'pocwisc3', 'pocwisc4', 'pocwisc5', 'pocwisc6', 'pocwisc7'], 
# 'pocwisc7', 'pocwisc8', 'pocwisc9', 'pocwisc10'],
#, 'output11', 'output12'], 
#'output1', 'output2','output3', 'output4', 'output5', 'output6', 'output7', 'output8', 'output9', 'output10'],
                                         job_type='',
                                         input_csv = 'CRF_DU_v2.csv',  #'CRF_full_data_corrected_v6.csv', # 'CRF_full_data_sample_size_train.csv',  #'CRF_full_data_corrected.csv',
                                         time_interval=0,
                                         n_split=7,
                                         shuffle = False, #True, #False, # False
                                         random_state=None,
                                         split_method= 'burnarea', # 'burnarea', #'session'
                                         keep_logs=False,
                                         gpic_selection_file= '/disk2/nik/Analyzer/test/pics.txt'
                                        )







