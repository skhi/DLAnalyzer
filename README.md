#  Deep Learning Analyzer

# setup environments for Caffe

export PYTHONPATH=/path/to/Caffe/python:$PYTHONPATH

# setup environments for the DLAnalyzer

export PYTHONPATH=/path/to/DLAnalyzer/python:$PYTHONPATH

export PYTHONPATH=/path/to/DLAnalyzer/config:$PYTHONPATH

#  Running DLAnalyzer

High-level instructions: 

  1. Go to the config directory and have a look at the *_template.py* files and the variables defined in it.
  2. To make your own configuration, go to *caffe_config.py* and *selector_config.py*, and have a look at the example configurations. This is a good starting point. 
  3. To run the DLAnalyzer, go to test directory. Here are some key information:
      
      **a.** csv file (CRF_data.csv in this example) which will include information about the patients (CRF), corresponding images    and a set of pictures, which will be used in the algorithm (both training and testing). In the csv file, the corresponding images are used as a .txt file, indicating the path to the iriginal raw images. 
      
      **b.** The actual data can be locating elsewhere
      
      **c.** The Segnet template model is located at: model_segnet. This is template and all variables in all three files will be set to those indicated in the config file. The template original directory is read only, so a copy with fixed values will be placed inside the working area.
      
      **d.** The output of the DLAnalyzer will be created in inside the *test* directory, named as you had indicated in the config file
      
 4. To run the machinery, execute *python run.py*   


export DYLD_LIBRARY_PATH=/usr/local/cuda/lib64:$DYLD_LIBRARY_PATH 
export PYTHONPATH=~/caffe-segnet-cudnn5/python:$PYTHONPATH
export PYTHONPATH=/disk2/nik/Analyzer/python:$PYTHONPATH
export PYTHONPATH=/disk2/nik/Analyzer/config:$PYTHONPATH
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda
