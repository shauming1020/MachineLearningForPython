# 1. Download YOLACT Model 
https://github.com/dbolya/yolact  
Follwing the Installation of Yolact step by step.
But don't install the pytorch using Anaconda, because we need to pytorch-gpu version but cpu version on Anaconda.

# 2. Install the pytorch with gpu version  
Key in the following instruction on Anaconda prompt:  
conda install pytorch cuda100 -c pytorch

# 3.Install COCO dataset on window10  
https://github.com/philferriere/cocoapi   
first, key in the following instruction on Anaconda prompt:  
conda install git  
then:  
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"  
