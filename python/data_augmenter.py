import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from logger import log
import sys


class ImageAugmenter:
    
        def __init__(self, npData, datasetImages, augType,  rotationRange, shiftRange, batchSize=1, imgHeight=360, imgWidth=480, seed=1):
            self.rotation_range = rotationRange
            self.shift_range    = shiftRange
            self.shift_range    = shiftRange
            self.data           = np.asarray(npData)
            self.images         = datasetImages
            self.batch_size     = batchSize
            self.height         = imgHeight
            self.width          = imgWidth
            self.augment_type   = augType
	    self.seed 		= seed
        
        def nullgen(self):
            datagen = ImageDataGenerator(featurewise_center=False,
                                         featurewise_std_normalization=False,
                                         dim_ordering='tf')
            return datagen
                
        def rotate(self):
            datagen = ImageDataGenerator(featurewise_center=False,
                                         featurewise_std_normalization=False,
                                         rotation_range=self.rotation_range,
                                         dim_ordering='tf') # maximum rotation is 90 degrees
            return datagen
        
        def shift(self):
            datagen = ImageDataGenerator(featurewise_center=False,
                                         featurewise_std_normalization=False,
                                         height_shift_range=self.shift_range,
                                         dim_ordering='tf') # width_shift_range=shift # width shift causes problems
            return datagen
        
        def flip(self):
            datagen = ImageDataGenerator(featurewise_center=False,
                                         featurewise_std_normalization=False,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         dim_ordering='tf')
            return datagen
        
        
        
        def aug(self, datagen, X, y):
            # fit parameters from data
            #            print 'ndim, ' , X.ndim
            #            print 'shape ', X.shape
            datagen.fit(X, seed=self.seed)
            
            # configure batch size
            datagen.flow(X, y, self.batch_size)
            
            # retrieve one batch of images
            X_batch, y_batch = datagen.next()
            XX = np.array(X_batch[0],'float32') # need to be float32, uint8 doesn't work
            
            return XX
        
        def pop(self, XX):
            # this function separates raw images and mask
            
            raw = XX[:,:,0:len(self.images)] # first 9 images = 8MSI + DeepView
            
            M = XX[:,:,len(self.images):] # last 3 images are mask

            # initialize data and label
            local_data  = np.empty(raw.shape,'uint8')
            local_label = np.empty(M.shape,'uint8')
            
            # make dim(raw)= H * W * 9 * 1, dim(mask)= H * W * 3 * 1
            local_data  = raw[..., np.newaxis]
            local_label = M[..., np.newaxis]
            
            return local_data, local_label
        
        def apply_augment(self):
            
            raw     = np.empty([self.height, self.width, len(self.images), 0],'uint8')
            classid = np.empty([self.height, self.width, 1, 0])
            

            #### prepare data for augmentation
            y = np.array([1], dtype = np.float32)
            X = self.data[np.newaxis, :]
            
            if len(self.augment_type) == 0:
                log.error("User provided no augmentation argument!")
            
            for type in self.augment_type:
            
                global XX
                
                if type == "default":
                    XX = self.aug(self.nullgen(), X, y)
                elif type == "rotation":
                    XX = self.aug(self.rotate(), X, y)
                elif type == "shift":
                    XX = self.aug(self.shift(), X, y)
                elif type == "flip":
                    XX = self.aug(self.flip(), X, y)
                else:
                    log.error("User provided wrong augmentation argument!")
                    sys.exit(0)

                (raw_, classid_) = self.pop(XX)
                r = np.empty((self.height, self.width, len(self.images), 1),'uint8')
                for i in range(raw_.shape[2]):
                    r[:,:,i,:] = raw_[:,:,i,:] / raw_[:,:,i,:].max() * 255
                raw       = np.append(raw, r, axis=3)
                classid_  = np.round(classid_)                
         	classid   = np.append( classid, classid_, axis=3)
            label = classid
        
            log.info( "datashape" + str(raw.shape) + ", data type " + str(raw.dtype) )
            log.info( "label shape" + str(label.shape) + ", label type " + str(label.dtype))
        
            return raw, label


