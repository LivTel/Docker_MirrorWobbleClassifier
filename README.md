In this application docker is being using purely to provide a controlled and sandboxed
runtime environment separate from the host. None of the classifier code, config
or data are inside the container. 

All the software runs from an external disk (/mnt/newarchive) 
which gets mounted to the container at runtime. Similarly we mount the incoming directory onto
the container so that it can access the new data as they arrive on lt-qc.

Ideally run on lt-qc because it has access to /data/incoming, but the old OS causes problems building the docker image.

The Classifier Model - NEEDS UPDATE
--------------------
The classifer is based on the vgg16 CNN image classifier. It was trained on about 600 each good and bad image
from mid 2020. The training was done using GPUs in the free online Google CoLab environment. Only the weights 
from the model were exported and are reimported here to use with the stock vgg16 model that is included in
python torchvision. 

The state_dict from the trained classifier is > 500Mb which seemed too large to include in teh github
repository. The copy we are actually using is installed on lt-qc in /mnt/newarchive1/VMshare/shutterclassifier/Models.
This needs to be backed up and maintained separatey. It is not in github.

This was all written for the IOO shutter classifier, but in principal the entire system can be used with
any other set of weights for vgg16 that have been trained to classify something else.


Installation
------------
On lt-qc
```shell
cd /mnt/newarchive1/Dockershare/
git clone https://github.com/LivTel/Docker_MirrorWobbleClassifier
cd Docker_MirrorWobbleClassifier
docker build -t mirrorwobbleclassifier .
docker run -id --name running_mirrorwobbleclassifier -v /mnt/newarchive1/Dockershare/Docker_MirrorWobbleClassifier/:/classifier/ -v /data/incoming:/data/incoming  mirrorwobbleclassifier
```
The container is now running and available for use.

Note that when the container was started, two directories from the host were mounted onto the container using -v
* The repository directory itself (/mnt/newarchive1/Dockershare/Docker_MirrorWobbleClassifier) is available in the container as /classifier
* The /data/incoming directory is available in the container as /data/incoming

Data Preparation - NEEDS UPDATE
----------------

The classifier was trained on 224x224 pixel FITS images created by 9x9 binning normal 2x2 binned IO:O images. 
Classification will be best if you feed it with similar data. It can work on larger FITS images but will not
be as accurate and, for example, is likely to be hopeless if you feed it full size 2048x2048 images.

Best to shrink the FITS images down first in some external application.

Included here is trivial python script that will boxcar median bin a FITS. Typically IO:O images are preprocessed by
```shell
python3 imbin.py -b 9 h_e_99999999_1_1_1_1.fits
```

Running Classification
----------------------
```shell
docker exec -it running_mirrorwobbleclassifier python3 /classifier/app_ml.py /data/incoming/h_e_99999999_1_1_1_1_bin9.fits
```

Running From Crontab
--------------------
Nothing to do with this classifier per se, but careful calling dockers from within cron scripts. As with many other
cron script you can have problems with the runtime environment. Do not use the -t pseudo-tty option. 
```shell
docker exec -i running_mirrorwobbleclassifier python3 /classifier/app_ml.py /data/incoming/h_e_99999999_1_1_1_1_bin9.fits
```
