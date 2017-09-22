import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('_', text) ]

alist=[
    "something_2_9.h5",
    "something_1_2.h5",
    "something_1_7.h5",
    "something_2_0.h5",
    "something_2_5.h5",
    "something_1_1.h5",
    "something_1_10.h5",
    "something_0_1.h5"]

alist_ = [e.split('.h5')[0] for e in alist]
print (alist_)
alist_.sort(key=natural_keys)
alist_ = [e+str('.h5') for e in alist_]
print(alist_)

net: "/home/Hinton/Documents/segnet_nik/segnet_jeff/model_segnet_bn/train.prototxt"     
test_initialization: false
test_iter: 1
test_interval: 100000
base_lr: 0.001 
lr_policy: "step"
gamma: 1.0
stepsize: 100000
display: 50
momentum: 0.9
max_iter: 4000
weight_decay: 0.0005
snapshot: 5000
snapshot_prefix: "/home/Hinton/Documents/segnet_nik/segnet_jeff/model_segnet_bn/segnets"    
solver_mode: GPU
