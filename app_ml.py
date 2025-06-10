# This is the shutter classifier script here as a placeholder
# Needs updating for the mirrorwobble


from timeit import default_timer as timer
preImports = timer()

import os, sys, argparse
import numpy as np
from astropy.io import fits

import torch
from torchvision import models
import torch.nn as nn

preArgparse = timer()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run shutter classifier on one image')

    parser.add_argument('infile', action='store', help='File to classify, mandatory')

    parser.add_argument('-d', dest='displayImage', action='store_true', help='Display the result as well as save FITS (default: Off)')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Turn on verbose mode (default: Off)')
    parser.add_argument('-t', dest='timing', action='store_true', help='Turn on timing reports (default: Off)')
    # -h is always added automatically by argparse1

    args = parser.parse_args()

if args.verbose:
  print (args)
  
if args.timing:
  print("Imports duration = ",preArgparse-preImports,"sec")
  print("Argparse duration = ",timer()-preArgparse,"sec")

#
# Download vgg16 and modifiy for our use
#
preModel = timer()
model = models.vgg16(pretrained=True)
#model.load_state_dict(torch.load('Models/vgg16-397923af.pth'))

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False

n_inputs = model.classifier[6].in_features
n_classes = 2
model.classifier[6] = nn.Sequential(
                      nn.Linear(n_inputs, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, n_classes),                   
                      nn.LogSoftmax(dim=1))
#model.classifier

# Mapping class name to model index
# This was set up in the training. Class names are not stored in the model dict.
# You just have to know what it was trained on. 
#
# vgg16-transfer-touse1.pt 
# My successful model trained in 2020 and used in all test and discussions
# [(1, 'classBad'), (0, 'classGood')]")
#
# vgg16-transfer-touse2.pt
# Retrain same model on same data with same settings on 2022-02-18
# Ought to be effectively identical to vgg16-transfer-touse1.pt
# [(0, 'classBad'), (1, 'classGood')]")
#
#model.load_state_dict(torch.load('Models/vgg16-transfer-touse1.pt'))
#model.load_state_dict(torch.load('/TestSet/Models/vgg16-transfer-touse1.pt',map_location=torch.device('cpu')))
model.load_state_dict(torch.load('/shutterclassifier/Models/vgg16-transfer-touse2.pt',map_location=torch.device('cpu')))

if args.timing:
  print("Model creation duration = ",timer()-preModel,"sec")

#if args.verbose:
#  total_params = sum(p.numel() for p in model.parameters())
#  print(f'{total_params:,} total parameters.')
#  total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#  print(f'{total_trainable_params:,} training parameters.')



# How it was trained. Do not need to know this here. Just evaluate the model.
#criterion = nn.NLLLoss()
#optimizer = optim.Adam(model.parameters())

preFits = timer()
#fitsname = "Bad/h_e_20200405_31_1_1_1_bin9.fits"
#fitsname = "Good/h_e_20200515_40_1_1_1_bin9.fits"
image = np.array( fits.open(args.infile, do_not_scale_image=False)[0].data, dtype=np.uint16)
if args.verbose:
  print("Type, Shape, Min, Max, Mean, Median, std in FITS",image.dtype, image.shape, image.min(), image.max(), image.mean(),np.median(image), image.std())
arcsinhimage = np.arcsinh(image)
pc = np.percentile(arcsinhimage,[0.1,1,50,90,99,100])
if args.verbose:
  print("pc arcsinhimage= ",pc)
black = pc[0]
white = pc[4]
image_norm8 = np.array( (np.clip(arcsinhimage,black,white) - black) * (255.000 / (white-black))  , dtype=np.uint8)
if args.verbose:
  print("Type, Shape, Min, Max, Mean, Median, std in Norm",image_norm8.dtype, image_norm8.shape, image_norm8.min(), image_norm8.max(), image_norm8.mean(),np.median(image_norm8), image_norm8.std())

# Centre Crop
naxis1 = image_norm8.shape[1]
naxis2 = image_norm8.shape[0]
trim_left = int(np.floor((naxis1 - 224 ) / 2))
trim_top = int(np.floor((naxis2 - 224 ) / 2))
if args.verbose:
  print("[process_image] gutter = ",trim_left, trim_top)
image_norm8 = image_norm8[trim_top:trim_top+224,trim_left:trim_left+224]
if args.verbose:
  print(image_norm8.shape)

# In the colab workbook I scaled to unity (0...1). 
# Note that torchvision transforms.ToTensor() does this automatically (in some cases!) 
# when loading tehimages into the trainer. Here we are just duplicating how
# the images were processed by transforms() for the the trainer.
# The name norm8 is a bit silly now. It is now a floating point, not 8bit int.
image_norm8 = image_norm8 / 255.0
if args.verbose:
  print("Type, Shape, Min, Max, Mean, Median, std in Norm",image_norm8.dtype, image_norm8.shape, image_norm8.min(), image_norm8.max(), image_norm8.mean(),np.median(image_norm8), image_norm8.std())

# Do nothing standardization
means = np.array([0.0, 0.0, 0.0]).reshape((3, 1, 1))
stds = np.array([1.00, 1.00, 1.00]).reshape((3, 1, 1))

image_3layer = (image_norm8 - means) / stds
if args.verbose:
  print(image_3layer.shape)

# Note we are here in 3*X*Y, like a tensor, not X*Y*3 like an image
if args.verbose:
  print("Mean in three layers of RGB (a)",image_3layer[0].mean(), image_3layer[1].mean(), image_3layer[2].mean())
  print("Std in three layers of RGB (a)",image_3layer[0].std(), image_3layer[1].std(), image_3layer[2].std())

image_tensor = torch.Tensor(image_3layer)

if args.verbose:
  print("predict, tensor shape = ",image_tensor.shape)

image_tensor = image_tensor.view(1, 3, 224, 224)

if args.timing:
  print("FITS handling duration = ",timer()-preFits,"sec")

preClassifier = timer()

with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(2)
        if args.verbose:
          print("out is ",out)
          print("ps is",ps)
          print("topk is",topk)
          print("topclass is",topclass)

if args.timing:
  print("Classifier duration = ",timer()-preClassifier,"sec")

if int(topclass[0][0]):
  className = "OK"
else:
  className = "Shutter"

print(int(topclass[0][0]),np.round(float(topk[0][0]),3), className, args.infile)

