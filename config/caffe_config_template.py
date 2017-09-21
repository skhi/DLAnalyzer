#!/usr/bin/env python
import collections

''' This is a configuration template for the caffe tool.'''

caffe_config_template = collections.namedtuple('caffe_config_template',
                                
                                [
                                 # path to the caffe directory
                                 'path_to_caffe',
                                 
                                 # path to caffe model template: in this directory, it is expected to be: inference.prototxt, train.prototxt, solver.prototxt files
                                 'path_to_model',
                                 
                                 # This work area will be created and the caffe output will go there, including predicted images and accuracy measurements: this path is with resepct to the current dir
                                 'caffe_work_area',
                                 
                                 # This is the working directory, where all output will be placed
                                 'output_work_area',
     
                                 ######  Below the config variables are Caffe related, and will parsed into the solver template file. The naming is identical to what you have in the solver file  ######
                                 
                                 #
                                 'test_initialization',
                                 
                                 #
                                 'test_iter',
                                 
                                 #
                                 'test_interval',
                                 
                                 #
                                 'base_lr',
                                 
                                 #
                                 'lr_policy',
                                 
                                 #
                                 'gamma',
                                 
                                 #
                                 'stepsize',
                                 
                                 #
                                 'display',
                                 
                                 #
                                 'max_iter',
                                 
                                 #
                                 'momentum',
                                 
                                 #
                                 'weight_decay',
                                 
                                 #
                                 'snapshot_prefix',
                                 
                                 #
                                 'snapshot',
                                 
                                 #
                                 'solver_mode',
                                 
                                 ######  End of the Caffe related config variables  ######
                                 
                                 
                                 # This is the path the HDF5 file which includes all training images.
                                 'train_dataset',
                                 
                                 # This is the path the HDF5 file which includes all test images.
                                 'test_dataset',
                                 
                                 # Batch size which goes in the inference prototxt file --> by default it is set to 1
                                 'test_batch_size',
                                 
                                 # Batch size which goes in the train prototxt file --> by default it is set to 1
                                 'train_batch_size',
                                 
                                 # This is the path to the pretrained weights for caffe. This can be used in case you don't need to re-train the algo
                                 'pretrained_weights',

                                 # This is the path to the trained model. This can be used in case you don't need to train the algo and already have the trained model for deployment
                                 'trained_weights',
                                 
                                 # This is file includes the log info from Caffe
                                 'caffe_log'
                                 ]
                                
                                )

# create an instance of the namedtuple template
caffe_config = caffe_config_template('','','', '',False,0,0,0,'',False,0,0,0,0,0,0,'','', '', '', 1, 1, '','','' )


## for cross-validation


caffe_cross_val_config_template = collections.namedtuple('caffe_cross_val_config_template',
                                               
                                               [
                                                # path to the caffe directory
                                                'path_to_caffe',
                                                
                                                # path to caffe model template: in this directory, it is expected to be: inference.prototxt, train.prototxt, solver.prototxt files
                                                'path_to_model',
                                                
                                                # This work area will be created and the caffe output will go there, including predicted images and accuracy measurements: this path is with resepct to the current dir
                                                'caffe_work_area',
                                                
                                                # This is the working directory, where all output will be placed
                                                'output_work_area',
                                                
                                                ######  Below the config variables are Caffe related, and will parsed into the solver template file. The naming is identical to what you have in the solver file  ######
                                                
                                                #
                                                'test_initialization',
                                                
                                                #
                                                'test_iter',
                                                
                                                #
                                                'test_interval',
                                                
                                                #
                                                'base_lr',
                                                
                                                #
                                                'lr_policy',
                                                
                                                #
                                                'gamma',
                                                
                                                #
                                                'stepsize',
                                                
                                                #
                                                'display',
                                                
                                                #
                                                'max_iter',
                                                
                                                #
                                                'momentum',
                                                
                                                #
                                                'weight_decay',
                                                
                                                #
                                                'snapshot_prefix',
                                                
                                                #
                                                'snapshot',
                                                
                                                #
                                                'solver_mode',
                                                
                                                ######  End of the Caffe related config variables  ######
                                                
                                                
                                                # This is the path the HDF5 file which includes all training images.
                                                'train_dataset',
                                                
                                                # This is the path the HDF5 file which includes all test images.
                                                'test_dataset',
                                                
                                                # Batch size which goes in the inference prototxt file --> by default it is set to 1
                                                'test_batch_size',
                                                
                                                # Batch size which goes in the train prototxt file --> by default it is set to 1
                                                'train_batch_size',
                                                
                                                # This is the path to the pretrained weights for caffe. This can be used in case you don't need to re-train the algo
                                                'pretrained_weights',
                                                
                                                # This is the path to the trained model. This can be used in case you don't need to train the algo and already have the trained model for deployment
                                                'trained_weights',
                                                
                                                # This is file includes the log info from Caffe
                                                'caffe_log'
                                                ]
                                               
                                               )

# create an instance of the namedtuple template
caffe_cross_val_config = caffe_cross_val_config_template('','',[''], [''],False,0,0,0,'',False,0,0,0,0,0,0,[''],'', '', '', 1, 1, '','','' )


