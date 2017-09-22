import sys
sys.path.append('/disk2/Faliu/caffe/python')
import caffe
import numpy as np
import os
import matplotlib.pyplot as plt


# init
caffe.set_device(int(0))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('../model/solver.prototxt')

niter=2000
display=50
test_iter=250
test_interval=50

_test_loss = 0

train_loss = np.zeros(niter // display)
test_loss = np.zeros(niter // test_interval)

for iter in range(niter):
    solver.step(1)
    if iter % display == 0:
        train_loss[iter // display]=solver.net.blobs['loss'].data
    if iter % test_interval == 0:
        for test_it in range(test_iter):
            solver.test_nets[0].forward()
            _test_loss+=solver.test_nets[0].blobs['loss'].data
        test_loss [iter // test_interval] = _test_loss / test_iter
        _test_loss= 0 



plt.plot(display*np.arange(len(train_loss)),train_loss,'r',label="Iter vs Loss: Training")
plt.plot(test_interval*np.arange(len(test_loss)),test_loss,'b',label="Iter vs Loss: Testing")
plt.legend(loc='upper left')
plt.savefig('iterloss.png')
plt.show()
