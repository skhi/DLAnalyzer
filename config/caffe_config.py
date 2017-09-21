#!/usr/bin/env python
from caffe_config_template import caffe_config


# General comment: Here you can define as many instancies of the "caffe_config" as you want. Each instance should be imported and then used separately in the run.py file. In this example, there is only one instance, called "test_caffe_config"

test_caffe_config = caffe_config._replace(path_to_caffe =  '/root/caffe-segnet-cudnn5/build/tools/caffe', #'/root/caffe-segnet-cudnn5/build/tools/caffe',
                                        path_to_model='/disk2/nik/Analyzer/test/model_segnet_mod_du', # model_segnet_final', #'/disk2/nik/Analyzer/test/model_segnet_bn',
                                        caffe_work_area='output/caffe',
				        output_work_area='output/prediction',
                                        test_initialization = 'false',
                                        test_iter=1,
                                        test_interval=10000000,
                                        base_lr=0.001,
                                        lr_policy='step',
                                        gamma=1.0,
                                        stepsize=10000000,
                                        display=20,
                                        max_iter='2000',
                                        momentum=0.9,
                                        weight_decay=0.0005,
                                        snapshot_prefix='output/training',
                                        snapshot=5000,
                                        solver_mode='GPU',
                                        trained_weights = '/disk2/Faliu/segNet/segNetFinal/segNetBNofficialCaffe/v3/snapshot/_iter_5000.caffemodel', #'/disk2/nik/Analyzer/test/output5/training_iter_5000.caffemodel',
					pretrained_weights = '/disk1/model_zoo/segNet/v2/segnet_pascal.caffemodel', 
                                        caffe_log ='caffe_log.txt'
                                        )

test_caffe_config_fcn = caffe_config._replace(path_to_caffe = '/root/caffe/build/tools/caffe',
                                        path_to_model='/disk2/nik/Analyzer/test/model_fcn_v6',
                                        caffe_work_area='output/caffe',
                                        output_work_area='output/prediction',
                                        test_initialization = 'false',
                                        test_iter=81,
                                        test_interval=999999999,
                                        base_lr=0.001,
                                        lr_policy='step',
                                        gamma=0.7,
                                        stepsize=100,
                                        display=50,
                                        max_iter='2000',
                                        momentum=0.9,
                                        weight_decay=0.005,
                                        snapshot_prefix='output/training',
                                        snapshot=200,
                                        solver_mode='GPU',
                                        trained_weights = '/disk2/nik/Analyzer/test/output_fcn_v6/training_iter_2000.caffemodel', #'/disk2/nik/Analyzer/test/output_fcn_pretrained/training_iter_2000.caffemodel',
                                        pretrained_weights = '/disk2/nik/Analyzer/test/vgg16-fcn.caffemodel',
                                        caffe_log ='caffe_log.txt'
                                        )


test_caffe_config_cnn = caffe_config._replace(path_to_caffe = '/root/caffe/build/tools/caffe',
                                        path_to_model='/disk2/nik/Analyzer/test/model_cnn',
                                        caffe_work_area='output/caffe',
                                        output_work_area='output/prediction',
                                        test_initialization = 'false',
                                        test_iter=500,
                                        test_interval=1000,
                                        base_lr=0.0001,
                                        lr_policy='step',
                                        gamma=0.7,
                                        stepsize= 1000,
                                        display=1000,
                                        max_iter='4000',
                                        momentum=0.9,
                                        weight_decay=0.0005,
                                        snapshot_prefix='output/training',
                                        snapshot=20000,
                                        solver_mode='GPU',
					train_batch_size = 80, 
                                        trained_weights = '/disk2/nik/Analyzer/test/output_cnn_new/training_iter_6037.caffemodel', # '/disk2/nik/Analyzer/test/output_cnn/training_iter_4000.caffemodel',
                                        pretrained_weights = '',
                                        caffe_log ='caffe_log.txt'
                                        )





