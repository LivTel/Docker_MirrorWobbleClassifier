from timeit import default_timer as timer
preImports = timer()

import os, sys, argparse, logging

debugLevel = logging.DEBUG
LOGFILE = '/mnt/external/classifier.log'
logging.basicConfig(level=debugLevel, filename=LOGFILE, format='%(asctime)s : %(name)s : %(message)s', encoding='utf-8')
# Log to STDOUT
#logging.basicConfig(level=debugLevel, stream=sys.stdout, format='%(asctime)s : %(name)s : %(message)s', encoding='utf-8')
# Set level on per library basis
#logging.getLogger('urllib3').setLevel(logging.WARNING)
#logging.getLogger('botocore').setLevel(logging.WARNING)
  


import socket, io
import numpy as np
from PIL import Image       # for JPG manipulation
#from astropy.io import fits

import torch
from torchvision import models
import torch.nn as nn

#HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
HOST = '0.0.0.0'
PORT = 8225        # Port to listen on (non-privileged ports are > 1023)

verbose = True

# Confidence required to accept a "Wobble" classification. Low confidence Wobbles are called OK
decisionThreshold = 0.6

debugLevel = logging.DEBUG
#logger = logging.getLogger(__name__)
# Log to file
#LOGFILE = 'classifier.log'
#logging.basicConfig(filename=LOGFILE, format='%(asctime)s : %(name)s : %(message)s', encoding='utf-8', level=debugLevel)
# Log to STDOUT
logging.basicConfig(level=debugLevel, stream=sys.stdout, format='%(asctime)s : %(name)s : %(message)s', encoding='utf-8')
# Set level on per library basis
#logging.getLogger('urllib3').setLevel(logging.WARNING)
#logging.getLogger('botocore').setLevel(logging.WARNING)


#def server():
#    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#      s.bind((HOST, PORT))
#      s.listen()
#
#      while True:
#        conn, addr = s.accept()
#        with conn:
#          print('Connected by', addr)
#        
#          # Receive file size
#          file_size_bytes = conn.recv(8)
#          file_size = int.from_bytes(file_size_bytes, 'big')
#
#          # Receive image data
#          image_data = b""
#          while len(image_data) < file_size:
#            chunk = conn.recv(4096)
#            if not chunk:
#              break
#            image_data += chunk
#        
#          # Save the received image
#          with open('received_image.jpg', 'wb') as f:
#            f.write(image_data)
#          print("Image received and saved as received_image.jpg")
#
#          #Send a response
#          conn.sendall( "The answer is 42".encode() )


if __name__ == '__main__':

  #
  # Download vgg16 and modifiy for our use
  #
  preModel = timer()
  logging.debug("Start loading classifier model" )
  model = models.vgg16(weights='VGG16_Weights.DEFAULT')

  # Freeze model weights
  for param in model.parameters():
    param.requires_grad = False

  n_inputs = model.classifier[6].in_features
  n_classes = 2
  model.classifier[6] = nn.Sequential(
                      nn.Linear(n_inputs, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.5),                  # Will be turned off when we call model.eval()
                      nn.Linear(256, n_classes),                   
                      nn.LogSoftmax(dim=1))
  logging.debug(model.classifier)

  # These were the trained classes
  # [(1, 'classBad'), (0, 'classGood')]")
  classes = ["Wobble", "OK"]

  # Load our model weights. Do not have these yet...
  logging.debug("Read classifier weights from external")
  #model.load_state_dict(torch.load('/mnt/external/Models/vgg16-transfer-wobble-20250609235857.pt',map_location=torch.device('cpu')))
  #model.load_state_dict(torch.load('/mnt/external/Models/vgg16-transfer-wobble-20250624013305.pt',map_location=torch.device('cpu')))
  #model.load_state_dict(torch.load('/mnt/external/Models/vgg16-transfer-wobble-20250627020015.pt',map_location=torch.device('cpu')))   # Model with most testing so far. Kaggle

  #fails model.load_state_dict(torch.load('/mnt/external/Models/vgg16-transfer-wobble-20250708T160639.pth',map_location=torch.device('cpu')))   # First Prospero test
  #torch.load('/mnt/external/Models/vgg16-transfer-wobble-20250708T160639.pth',map_location=torch.device('cpu'),weights_only=False)   # First Prospero test
  model.load_state_dict(torch.load('/mnt/external/Models/vgg16-transfer-wobble-nll-20250708T211850.pt',map_location=torch.device('cpu')))  

  #if verbose:
  #  total_params = sum(p.numel() for p in model.parameters())
  #  print(f'{total_params:,} total parameters.')
  #  total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  #  print(f'{total_trainable_params:,} training parameters.')

  # How it was trained. Do not need to know this here. Just evaluate the model.
  #criterion = nn.CrossEntropyLoss() 
  #optimizer = optim.Adam(model.parameters())

  logging.debug("Model creation duration = %f sec",timer()-preModel)

  # Start waiting for images to arrive on socket
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    if verbose:
      print("Now waiting for JPEGs from a client")

    while True:

      conn, addr = s.accept()
      with conn:
        preImageXfer = timer()
        logging.info('Connected by %s', addr)
        
        # Receive file size
        file_size_bytes = conn.recv(8)
        file_size = int.from_bytes(file_size_bytes, 'big')
        logging.debug('Filesize %ld', file_size)

        # Receive image data
        image_data = b""
        while len(image_data) < file_size:
          chunk = conn.recv(4096)
          if not chunk:
            break
          image_data += chunk
          #logging.debug('Read %d chunk. Total so far got %ld', len(chunk), len(image_data))

        logging.debug("JPG size %d",len(image_data))

        image_stream = io.BytesIO(image_data)
        img = Image.open(image_stream)
        logging.debug(img.info)
        logging.debug('PIL image object: %s', img)
        
        # Save the received image
        #with open('received_image.jpg', 'wb') as f:
        #  f.write(image_data)
        # Successfully tested that img is a good image.
        #img.save('received_image.jpg')
        #print("Image received and saved as received_image.jpg")

        img_rgb = img.convert('RGB')
        rgb_array = np.array(img_rgb)
        # Need to copy it into an RGB array
        logging.debug("numpy array shape: %s",rgb_array.shape)

        logging.debug("Image transfer duration = %f sec",timer()-preImageXfer)

        preImageProc = timer()
        # On the shutter classifer this required binning, scaling and conversion to tiff.
        # This is much simpler. We cannot bin becise it loses too much detail and JMM has already done the scale and conversion to JPEG for us.

        # For training I scaled to unity (0...1) as floats. 
        # Note that torchvision transforms.ToTensor() does this automatically (in some cases!) when loading the images into the trainer. Here we are just duplicating how
        # the images were processed by transforms() for the the trainer.
        rgb_array = (rgb_array / 255.0).astype(np.float32)
        logging.debug("Type, Shape, Min, Max, Mean, Median, std in Norm %s %s %f %f %f %f %f",rgb_array.dtype, rgb_array.shape, rgb_array.min(), rgb_array.max(), rgb_array.mean(),np.median(rgb_array), rgb_array.std())

        # Do nothing standardization
        #means = np.array([0.0, 0.0, 0.0]).reshape((3, 1, 1))
        #stds = np.array([1.00, 1.00, 1.00]).reshape((3, 1, 1))
        #image_3layer = (image_norm8 - means) / stds
        image_3layer = rgb_array
        logging.debug("image_3layer.shape: %s", image_3layer.shape)

        # Currently an X*Y*3 like an image
        logging.debug("Mean in three layers of RGB %f %f %f",image_3layer[:,:,0].mean(), image_3layer[:,:,1].mean(), image_3layer[:,:,2].mean())
        logging.debug("Std in three layers of RGB %f %f %f",image_3layer[:,:,0].std(), image_3layer[:,:,1].std(), image_3layer[:,:,2].std())
        # All numbers are identical to those I get in kaggle, so the data are read and handled correctly

        # Torch needs us to reshape the array to 3*X*Y for the tensor (not X*Y*3 like an image)
        image_tensor = torch.from_numpy(image_3layer).permute(2, 0, 1)
        logging.debug("tensor shape: %s",image_tensor.shape)

        # Convert from a single tensor to an array of 1 tensor!
        image_tensor = image_tensor.view(1, image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2])
        logging.debug("tensor shape: %s",image_tensor.shape)

        logging.debug("Image handling duration = %f sec",timer()-preImageProc)

        preClassifier = timer()

        # no_grad() tells it not to bother calculating the gradients. We do not need them for need evaluation, only training.
        with torch.no_grad():
          # Place in evaluate mode, disables DropOut and other training features.
          model.eval()
          # Model outputs log probabilities
          out = model(image_tensor)
          ps = torch.exp(out)
          to_p, to_class = ps.topk(2, largest=True, sorted=True)       # Get the N largest probabilities
          best_p = to_p.numpy()[0]
          best_class = to_class.numpy()[0]

        logging.debug("Classifier duration = %f sec",timer()-preClassifier)

        logging.debug("P %f, Class %d",best_p[0], best_class[0])

        # Provisional decision is the best (first) probability
        decision = classes[best_class[0]]
        decision_prob = best_p[0]

        # But if the bad confidence is below threshold, default to OK and we need the associated probabilty
        if decision == "Wobble" and best_p[0] < decisionThreshold :
          decision = "OK"
          decision_prob = best_p[0]


        logging.info("Highest prob = %s (%f), Final decision = %s (%f)", classes[best_class[0]], best_p[0], decision, decision_prob)

        #Send a response
        conn.sendall( (decision+" "+str(np.round(decision_prob,3)) ).encode() )
