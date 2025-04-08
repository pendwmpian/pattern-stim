# 
# This is the code for align the FOV detection
#

# import libraries
import socket
import serial
import numpy as np
import time
import os
import sys
import datetime
import random
import cv2
import threading
import oasis_camera


# define TCP server address
host = "localhost"
port_stim = 2222
port_camera = 2226

# camera parameter 
FOV_RADIUS = 414

# Convert 8-bit grayscale image to 1-bit black-white image
def Convert(img8):
	s=img8.shape
	a=img8.reshape(s[0]*s[1]//8, 8)
	a2 = np.ones(a.shape)
	for i in range(7):
		a2[:, i] = 1 << (7 - i)
	a = a*a2
	return a.sum(axis=1)

def FormatImage(img):
    width = img.shape[1]
    img = ((img > 127) * np.ones(img.shape)).astype(np.uint8)
    n = width % 8
    if n != 0:
        pad = np.zeros((img.shape[0], 8 - n), dtype=np.uint8)
        img = np.hstack((img, pad))
    return Convert(img).astype(np.uint8)

def detectOuterCircle(image, radius=FOV_RADIUS):

    image = image >> 8
    image = image.astype(np.uint8)

    cv2.imwrite('./data/alignment/0_original.jpg', image)
    blurred = cv2.GaussianBlur(image, (9, 9), 1.5)
    cv2.imwrite('./data/alignment/1_blurred.jpg', blurred)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=60)
    cv2.imwrite('./data/alignment/2_edges.jpg', edges)
    
    circle_template = np.zeros((radius * 2, radius * 2), dtype=np.uint8)
    cv2.circle(circle_template, (radius, radius), radius, 255, 2)

    result = cv2.matchTemplate(edges, circle_template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    center_x, center_y = max_loc[0] + radius, max_loc[1] + radius

    output_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.circle(output_image, (center_x, center_y), radius, (0, 255, 0), 2)
    cv2.imwrite('./data/alignment/3_circle_edge.jpg', output_image)
    return center_x, center_y, radius


s_camera = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
remote_ip_camera = socket.gethostbyname( host )
s_camera.connect((remote_ip_camera, port_camera))

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
remote_ip = socket.gethostbyname( host )
s.connect((remote_ip, port_stim))

pattern_img = np.ones((200, 200), dtype=np.uint8) * 255
w = np.uint32(200); h = np.uint32(200)

func = np.uint32(2)
s.send(func)
w = np.uint32(w)
s.send(w)
h = np.uint32(h)
s.send(h)
s.send(pattern_img)

# Receive Camera Image
oasis_camera.RequestAllImages(s_camera)
oasisReadImage = oasis_camera.ReadImage(s_camera)
oasis_image = oasisReadImage.receive()
fov_center_x, fov_center_y, _ = detectOuterCircle(oasis_image)

print(fov_center_x, fov_center_y, sep=" ")

# # define the pattern image size
# w = 200; h = 200
# half_patt_border_x = w * fov_center_x // oasisReadImage.width()
