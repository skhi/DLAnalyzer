#!/usr/bin/env python
from caffe_config_template import caffe_cross_val_config


# General comment: Here you can define as many instancies of the "caffe_config" as you want. Each instance should be imported and then used separately in the run.py file. In this example, there is only one instance, called "test_caffe_config"

test_caffe_cross_val_config = caffe_cross_val_config._replace(path_to_caffe =  '/root/caffe-segnet-cudnn5/build/tools/caffe',
                                                    path_to_model='/disk2/nik/Analyzer/test/model_segnet_final',
                                                    caffe_work_area=['pocwisc1/caffe','pocwisc2/caffe','pocwisc3/caffe','pocwisc4/caffe','pocwisc5/caffe', 'pocwisc6/caffe', 
								     'pocwisc7/caffe'], 
  # 'pocwisc8/caffe', 'pocwisc9/caffe'], #'pocwisc10/caffe'],
                                                                    # 'output11/caffe',
                                                                    # 'output12/caffe'],
								    # 'output4/caffe', 
								     #'output5/caffe', 
								     #'output6/caffe', 'output7/caffe', 'output8/caffe', 'output9/caffe', 'output10/caffe'],
                                                    output_work_area=['pocwisc1/prediction','pocwisc2/prediction','pocwisc3/prediction','pocwisc4/prediction',
                                                                      'pocwisc5/prediction','pocwisc6/prediction',
                                                                      'pocwisc7/prediction'], #'pocwisc8/prediction', 'pocwisc9/prediction' ],
								     # 'pocwisc9/prediction', 'pocwisc10/prediction'],
                                                                     # 'output11/prediction',
                                                                     # 'output12/prediction'], 
    #'output4/prediction', 'output5/prediction', 'output6/prediction',
#								      'output7/prediction', 'output8/prediction', 'output9/prediction', 'output10/prediction'],
                                                    test_initialization = 'false',
                                                    test_iter=1,
                                                    test_interval=10000000,
                                                    base_lr=0.001,
                                                    lr_policy='step',
                                                    gamma=1.0,
                                                    stepsize=10000000,
                                                    display=20,
                                                    max_iter='5000',
                                                    momentum=0.9,
                                                    weight_decay=0.0005,
                                                    snapshot_prefix=['pocwisc1/training','pocwisc2/training','pocwisc3/training','pocwisc4/training',
                                                                     'pocwisc5/training','pocwisc6/training', 'pocwisc7/training'], # 'pocwisc8/training','pocwisc9/training'],
                                                                   #   'pocwisc7/training', 'pocwisc8/training', 
 								   #   'pocwisc9/training', 'pocwisc10/training'],
                                                                   #  'output11/training',
                                                                   #  'output12/training'], 
#'output4/training','output5/training', 'output6/training', 
#						                     'output7/training','output8/training', 'output9/training', 'output10/training'],
                                                    snapshot=5000,
                                                    solver_mode='GPU',
						    train_batch_size = 4,
                                                    trained_weights = '/disk2/nik/Analyzer/test/training_iter_5000.caffemodel', #'/disk2/nik/Analyzer/test/FaliuModels/v10/test_weights.caffemodel',
                                                    pretrained_weights = '/disk1/model_zoo/segNet/v2/segnet_pascal.caffemodel',
                                                    caffe_log ='caffe_log.txt'
                                                    )




