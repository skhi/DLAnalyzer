import os, sys
from os import chdir, listdir
import shutil
import glob
import subprocess
import caffe
import numpy as np
import scipy.misc
import scipy.io
from logger import log
import pandas_ml as pdml
import fileinput
from PIL import Image
from itertools import product

import threading

class CaffeRunner_Cross_Val:
    '''Caffe class'''

    def __init__(self, namedtupleConfig, n_fold):

        self.fold                = n_fold
        self.path_to_caffe       = namedtupleConfig.path_to_caffe
        self.path_to_model       = namedtupleConfig.path_to_model
        self.work_area           = namedtupleConfig.caffe_work_area[self.fold]
	self.output_area         = namedtupleConfig.output_work_area[self.fold]
        self.train_dataset       = namedtupleConfig.train_dataset
        self.train_batch_size    = namedtupleConfig.train_batch_size
        self.test_dataset        = namedtupleConfig.test_dataset
        self.test_batch_size     = namedtupleConfig.test_batch_size
	self.caffe_log           = namedtupleConfig.caffe_log
        self.namedtupleConfig    = namedtupleConfig
        self.current_dir         = os.getcwd()
        self.solver_file         = ''
        self.test_prototxt       = ''
        self.train_prototxt      = ''
        
        
        # create work areas
        
        if self.work_area not in self.namedtupleConfig.snapshot_prefix[self.fold]:
            self.__create_workarea(self.work_area)
            self.__create_subworkarea(self.namedtupleConfig.snapshot_prefix[self.fold])
        else:
            self.__create_subworkarea(self.namedtupleConfig.snapshot_prefix[self.fold])

	# creating a directory for the output pictures
	self.__create_subworkarea(self.output_area)
    

    
    
    def __create_workarea(self, area):
        if os.path.isdir(area):
            log.error("Work "+str(area)+" area already exists!")
            exit()
        else:
            os.mkdir(area)

    def __create_subworkarea(self, area):
        if os.path.isdir(area):
            log.error("Sub-work "+str(area)+" area already exists!")
            exit()
        else:
            log.info( "subdirectory tree ")
            os.makedirs(area)



    def __copy_model(self, path_to_model, dest):
        shutil.copytree(path_to_model, dest)

    def prepare_config(self):

        self.__copy_model(self.path_to_model, self.work_area+os.sep+self.path_to_model.split('/')[-1])
        chdir(self.work_area+os.sep+self.path_to_model.split('/')[-1])
        # look for a file for training config: For example "segnet_solver.prototxt"
        file_name = glob.glob("*prototxt")
        
        if len(file_name) != 3:
            log.error("There are less/more prototxt files than expected, check what's going on!")
            sys.exit(0)
        
        # sort the list alphabetically
        file_name.sort()
        
        # open *inference file and propagate config parameters
        log.info("Opening inference file: " +str(file_name[0]))
        with open(file_name[0], 'r') as f:
            filedata = f.read()
        
        filedata = filedata.replace("test_dataset", str(self.test_dataset))
        filedata = filedata.replace("test_batch_size", str(self.test_batch_size))

        with open(file_name[0], 'w') as f:
            f.write(filedata)
        
        self.test_prototxt = os.getcwd() + os.sep + file_name[0]

        # open *train file and propagate config parameters
        log.info("Opening training file: " +str(file_name[2]))
        with open(file_name[2], 'r') as f:
            filedata = f.read()
        
        filedata = filedata.replace("train_dataset", str(self.train_dataset))
        filedata = filedata.replace("train_batch_size", str(self.train_batch_size))
        
        with open(file_name[2], 'w') as f:
            f.write(filedata)

        self.train_prototxt = os.getcwd() + os.sep + file_name[2]


        # open *solver file and propagate config parameters
        lines=[]
        log.info("Opening solver file: " +str(file_name[1]))
        with open(file_name[1], 'r') as f:
            for line in f:
                if "net:" in line :
                    lines.append( line.rstrip('\n').replace("NET", self.train_prototxt)+str('\n') )
                for key, value in self.namedtupleConfig._asdict().iteritems():
                    line_key = line.split(':')[0]
		    if key == line_key and line_key == "snapshot_prefix":
                        lines.append( line.rstrip('\n').replace(key.upper(), str(value[self.fold]))+str('\n') )
                    elif key == line_key:
                        lines.append( line.rstrip('\n').replace(key.upper(), str(value))+str('\n') )


        with open(file_name[1], 'w') as f:
            f.writelines(lines)
                
        log.info("Caffe runs with this configuration\n" + "".join(lines))
        self.solver_file = os.getcwd() + os.sep + file_name[1]

        chdir(self.current_dir)

    def get_caffemodel(self):
	pieces = self.namedtupleConfig.snapshot_prefix[self.fold].split('/')[:-1]
	path   = "/".join(pieces)
        names = glob.glob( path + "/*caffemodel")
        log.info("caffe models found\n" + "".join(names))
        return names    

    def train(self, num_gpu = 0):
        log.info("caffe training step")
        bashcmd = self.path_to_caffe + ' train -gpu '+str(num_gpu)+' -solver ' + self.solver_file + ' 2>&1 | tee ' + self.caffe_log
        log.info("Calling the following command for training:\n" + bashcmd)
        run_output=""
        try:
            run_output= subprocess.check_output(bashcmd, shell=True)
            log.info(run_output)
        except subprocess.CalledProcessError as e:
            log.info("The following output error:\n" + e.output)

    def re_train(self, num_gpu = 0):
        log.info("caffe training step")
        bashcmd = self.path_to_caffe + ' train -gpu '+str(num_gpu)+' -solver ' + self.solver_file + ' -weights '  + self.namedtupleConfig.pretrained_weights + ' 2>&1 | tee ' + self.caffe_log
        log.info("Calling the following command for training:\n" + bashcmd)
        run_output=""
        try:
            run_output= subprocess.check_output(bashcmd, shell=True)
            log.info(run_output)
        except subprocess.CalledProcessError as e:
            log.info("The following output error:\n" + e.output)


    def re_train_threaded(self, num_gpu=0):
        threading.Thread(target=self.re_train(num_gpu)).start()

    def test(self, imageNumber, imageHeight=360, imageWidth=480, use_trained_weights=False):
        '''
        Following batch normalization, a caffe neural network model can be used to make predictions
        on a set of unknown data. This function will take a neural network model defined as an
        'inference' model, such as: segnet_inference_9chan.prototxt.  in addition the trained weights
        are supplied to the fucntion.
        
        PLEASE NOTE: If error occurs, such as kernel shutdown, make sure the directory
        to the hdf5 database, '___.txt', is correctly identified in the 'inference_model.prototext' file.
        
        ### Input Parameters
        # hdf5 database: The location of the hdf5 database, for instance 'test h5', is specified in the inferenceModel.prototext file
        # inferenceModel: inference model
        # weights: CNN network weights as a result of the function compute_bn_statistics
        # resultsDir: results saved to this directory
        # imageHeight, imageWidth: height and width, used to calculate total number of pixels for average accuracy calculation
        '''

        cafee_model = ''
        if use_trained_weights is False:
            weights = self.get_caffemodel()
            if len(weights) == 0:
                log.info("Caffe model is not there!")
                sys.exit()
	    log.info("Caffe model found: "+str(weights[0])+"")
            cafee_model =  weights[0]
        else:
            cafee_model = self.namedtupleConfig.trained_weights
	    log.info("Caffe model used: "+str(cafee_model)+"")


        iter   = imageNumber  #: number of images in the hdf5 database that will be processed by this function
        txtloc = self.output_area + "/confusion_matrix.txt" #  The accuracy log will be saved to this location
        
        caffe.set_mode_gpu() # set using GPU, if use CPU, call caffe.set_mode_cpu()
        
        net = caffe.Net(self.test_prototxt, cafee_model, caffe.TEST) # load net
        fid = open(txtloc, 'w+') # open accuracy log
        
	fid.write( "pandas implementation of Confusion Matrix\n")
        for i in range(iter):
            
            net.forward() # everytime net forward is called, one batch of output is produced. In our case, batch=1 in inference model
            
            image = net.blobs['data'].data
            label = net.blobs['label'].data
            predicted = net.blobs['prob'].data # probability, number of channels is euqal to number of outputs in the model
            
            image = np.squeeze(image[0,:,:,:])
            image = image.transpose((1,2,0)) # change data dimension from channel*H*W to H*W*channel
            image = image / 255
            image = image[:,:,(2,1,0)] # switch RGB channel
            
            # display image
            #BM-no-plt plt.imshow(image)
            #BM-no-plt plt.figure(i + 1)
            
            output = np.squeeze(predicted[0,:,:,:]) # squeeze 4D to 3D
            ind    = np.argmax(output, axis = 0)
            label  = np.squeeze(label)
            
            ind    = ind.astype('uint8')
            label  = label.astype('uint8')
            
            fid.write( str('=' *30 + ("image %d:" %i ) + '='*30)+'\n' )
            
            log.info("uniques of label "+str(np.unique(label.ravel())))
            if len( np.unique(label.ravel()) ) == 1:
                log.info("Image_"+str(i)+" has only one class - no confusion matrix is calculate!")
            else:
                pdml_cm = pdml.ConfusionMatrix( label.ravel(), ind.ravel() )
                sys.stdout = fid
	        print pdml_cm.print_stats()
        #    sys.stdout = sys.__stdout__
            # save Result, Mask and Image
            scipy.misc.imsave(self.output_area + "/raw_" + str(i) + '.tif', image)
            scipy.misc.imsave(self.output_area + "/prediction_" + str(i) + '.tif', ind.astype(float)) # convert to float to save visible mask
            scipy.misc.imsave(self.output_area + "/mask_" + str(i) + '.tif', label.astype(float))
            scipy.io.savemat(self.output_area + "/probability_" + str(i) + '.mat', dict(img=predicted[0,1,:,:]) )
        fid.close()

    def test_cnn(self, imageNumber, hdf5Number, patchNumber, imageHeight=360, imageWidth=480, use_trained_weights=False):

            
        cafee_model = ''
        if use_trained_weights is False:
            weights = self.get_caffemodel()
            if len(weights) == 0:
                log.info("Caffe model is not there!")
                sys.exit()
            log.info("Caffe model found: "+str(weights[0])+"")
            cafee_model =  weights[0]
        else:
            cafee_model = self.namedtupleConfig.trained_weights
            log.info("Caffe model used: "+str(cafee_model)+"")
            

        txtloc = self.output_area + "/confusion_matrix.txt" #  The accuracy log will be saved to this location
                                                                
        caffe.set_mode_gpu() # set using GPU, if use CPU, call caffe.set_mode_cpu()
                                                                    
        net = caffe.Net(self.test_prototxt, cafee_model, caffe.TEST) # load net
        fid = open(txtloc, 'w+') # open accuracy log
        fid.write( "pandas implementation of Confusion Matrix\n")

        for image_i in range(imageNumber):
            pInfer = []
            lInfer = []
	    dInfer = []
	    rawImg = np.zeros((8,imageHeight,imageWidth))
            pixels = 0 
            for hdf5, patch in product(range(hdf5Number), range(patchNumber)):
                if pixels >= (imageHeight * imageWidth):
                   break
		    		
                net.forward() # everytime net forward is called, one batch of output is produced. In our case, batch=1 in inference model

                image     = net.blobs['data'].data
                label     = net.blobs['label'].data
                predicted = net.blobs['prob'].data # probability, number of channels is equal to number of outputs in the model

                             
                # raw images
                rawImgPatch = np.squeeze(image[0,:,:,:])
                rawImgPixel = rawImgPatch[:,17,17]

                row = pixels // imageWidth
                col = pixels % imageWidth
                rawImg[:,row,col] = rawImgPixel
                
                # prediction images
                pInfer.append(np.argmax( np.array(predicted[0])))

                # mask images
                lInfer.append(np.array(label[0]))

                pixels = pixels+1
 
	    # raw images
	    rawImg = rawImg.transpose((1,2,0))
	    rawImg = rawImg / 255
            rawImg = rawImg[:,:,(2,1,0)]
            rawImg = rawImg[17:-16,17:-16,:]
	    scipy.misc.imsave(self.output_area + "/raw_" + str(image_i) + '.tif', rawImg)
        
	    # predicted images
            predImg  = np.reshape(pInfer, (imageHeight, imageWidth))
	    predImg  = predImg[17:-16,17:-16]
	    predImg2=Image.fromarray(np.uint8(255*(np.array(predImg))))
            predImg2.save(self.output_area+ "/prediction_" + str(image_i) + ".png")

	    # mask images
	    labelImg = np.reshape(lInfer, (imageHeight, imageWidth))
	    labelImg = labelImg[17:-16,17:-16]
            labelImg2=Image.fromarray(np.uint8(255*(np.array(labelImg))))
            labelImg2.save(self.output_area+ "/mask_" + str(image_i) + ".png")
                                                 
            pdml_cm = pdml.ConfusionMatrix( labelImg.ravel(), predImg.ravel() )
            sys.stdout = fid
            print pdml_cm.print_stats()
            fid.close()
       






	









