import sys
sys.path.append('/disk2/Faliu/caffe/python')
import caffe

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import h5py

pInfer=[]
path="/disk2/Faliu/cnn/data/infer/inference"
for root,dirs,files in os.walk(path):
    for name in files:
        if name.endswith('.h5'):        
            filename=root+"/"+name
            print(filename)
            f=h5py.File(filename,'r')
            in_=list(f['data'])
            for iter in range(len(in_)):
                img=in_[iter]
                #label=list(f['label'])[iter]
                
                # load net
                net = caffe.Net('../model/deploy.prototxt', '../snapshot/_iter_6000.caffemodel', caffe.TEST)
                # shape for input (data blob is N*C*H*W)
                net.blobs['data'].reshape(1,*img.shape)
                net.blobs['data'].data[...]=img
                
                # run net and take prob for prediction
                net.forward()
                out=np.array(net.blobs['prob'].data[0])
                pred=np.argmax(out)
                pInfer.append(pred)

print(pInfer)
print('=====================================================================')
predImg = np.reshape(pInfer, (360, 480))
print(predImg)

predImg2=Image.fromarray(np.uint8(255*(np.array(predImg))))
predImg2.show()

predImg2.save('test222.png')
            
            
#            im=Image.open(filename)
#            in_ = np.array(im, dtype=np.float32)
#            in_ = in_[:,:,::-1]
#            in_ -= np.array((141.1247,157.8365,99.4236))
#            in_ = in_.transpose((2,0,1))
#            
#            # load net
#            net = caffe.Net('deploy.prototxt', 'snapshot/_iter_10000.caffemodel', caffe.TEST)
#            
#            # shape for input (data blob is N x C x H x W), set data
#            net.blobs['data'].reshape(1, *in_.shape)
#            net.blobs['data'].data[...] = in_
#            # run net and take argmax for prediction
#            net.forward()
#            out = net.blobs['score'].data[0].argmax(axis=0)
#            
#            out2=Image.fromarray(np.uint8(255*out))
#            
#           # plt.imshow(out2)
#           # plt.show()
#            newName=name[0:-4]+'_mask.png'
#            newfilename=root+"/"+newName
#            #print(newfilename)
#            out2.save(newfilename)
