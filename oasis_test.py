import numpy as np
import cv2
import socket

# Create rotating bar
def Convert(img8):
    s=img8.shape
    a=img8.reshape(s[0]*s[1]//8, 8)
    a2 = np.ones(a.shape)
    for i in range(7):
        a2[:, i] = 1 << (7 - i)
    a = a*a2
    return a.sum(axis=1)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
remote_ip = socket.gethostbyname("localhost")
s.connect((remote_ip, 2222))

width, height = 635, 417
cen = [300, 300]  # sample

testImg = np.zeros((height, width), dtype=np.uint8)
for i in range(50):
    for k in range(50):
        try:
            testImg[cen[1] - 25 + i, cen[0] -25 + k] = 255
        except IndexError:
            pass
testImg = Convert(testImg).astype(np.uint8)

func = np.uint32(2)
s.send(func)
w = np.uint32(width)
s.send(w)
h = np.uint32(height)
s.send(h)
s.send(testImg)
