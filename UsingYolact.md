# 1. Download YOLACT Model 
https://github.com/dbolya/yolact  
Follwing the Installation of Yolact step by step.
But don't install the pytorch using Anaconda, because we need to pytorch-gpu version but cpu version on Anaconda.

# 2.Download COCO dataset  
Image data  
Downloading MSCOCO train images ...  
http://images.cocodataset.org/zips/train2017.zip  
Downloading MSCOCO val images ..."
http://images.cocodataset.org/zips/val2017.zip   
Unzip to ".\yolact-master\data\coco\images"

Download the annotation data.  
echo "Downloading MSCOCO train/val annotations ..."
http://images.cocodataset.org/annotations/annotations_trainval2014.zip  
http://images.cocodataset.org/annotations/annotations_trainval2017.zip   
Unzip to ".\yolact-master\data\coco\annotations"  

# 3. Install the pytorch with gpu version  
Key in the following instruction on Anaconda prompt:  
conda install pytorch cuda100 -c pytorch  

# 4. Install the COCO API  
https://github.com/philferriere/cocoapi   
first, key in the following instruction on Anaconda prompt:  
conda install git  
then:  
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"  
