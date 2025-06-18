import socket
import os
import argparse

#HOST = '150.204.240.157'
HOST = '127.0.0.1'
PORT = 8225
JPEG_NAME = '/mnt/Dockershare/mirrorwobbleclassifier/TrainingData/Bad/h_e_20240107_104_1_1_1.jpg'           # Bad
#JPEG_NAME = '/mnt/Dockershare/mirrorwobbleclassifier/TrainingData/Bad/h_e_20241003_205_2_1_1.jpg'           # Good

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run M1 wobble classifier on one image')

    parser.add_argument('infile', action='store', help='JPG icreated by fits2grey, mandatory')

    #parser.add_argument('-d', dest='displayImage', action='store_true', help='Display the result as well as save FITS (default: Off)')
    #parser.add_argument('-v', dest='verbose', action='store_true', help='Turn on verbose mode (default: Off)')
    #parser.add_argument('-t', dest='timing', action='store_true', help='Turn on timing reports (default: Off)')
    # -h is always added automatically by argparse1

    args = parser.parse_args()


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    
    # Open image in binary mode
    with open(args.infile, 'rb') as f:
        image_data = f.read()

    # Send file size
    file_size = len(image_data)
    file_size_bytes = file_size.to_bytes(8, 'big')
    s.sendall(file_size_bytes)

    # Send image data
    s.sendall(image_data)
    #print("Image sent.")

    reply = s.recv(1024) 		# Receive up to 1024 bytes
    #if not reply:
    #  break # Client disconnected

    print(reply.decode())


