# import libraries
import socket
import numpy as np
import time
import os
import sys
import datetime
import random
import cv2
from oasis_behavioral_task1 import oasis_camera


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

# stimulation parameters

STIM_DURATION = 40 # msec
INTERVAL_DURATION = 160 # msec

LOGFILR_DIR = './logs'

FOV_RADIUS = 414
STIM_EDGE = 500

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

def generate_image_sequence(half_patt_border_x):
    """
    generate 4 patterns
    0: off
    1: full
    2: left
    3: right
    """

    pattern_seq = []
    print(half_patt_border_x)

    # Image 0 (off)
    img = np.zeros((STIM_EDGE, STIM_EDGE), dtype=np.uint8) * 255
    pattern_seq.append(FormatImage(img))

    # Image 1 (full exposure)
    img = np.ones((STIM_EDGE, STIM_EDGE), dtype=np.uint8) * 255
    pattern_seq.append(FormatImage(img))

    # Image 2 (half exposure: left)
    half_black = np.zeros((STIM_EDGE, STIM_EDGE - half_patt_border_x), dtype=np.uint8)
    half_white = np.ones((STIM_EDGE, half_patt_border_x), dtype=np.uint8) * 255

    img = np.hstack((half_white, half_black))
    # img = img.transpose()  # if needed
    pattern_seq.append(FormatImage(img))

    # Image 3 (half exposure: left)
    half_black = np.zeros((STIM_EDGE, half_patt_border_x), dtype=np.uint8)
    half_white = np.ones((STIM_EDGE, STIM_EDGE - half_patt_border_x), dtype=np.uint8) * 255
    
    img = np.hstack((half_black, half_white))
    # img = img.transpose()  # if needed
    pattern_seq.append(FormatImage(img))

    return pattern_seq


def generate_stimulation_timing(ntime, duration):
    """
    ntime (int): the number of stimulation sequences per 1min
    duration (int): stimulation duration (min)
    """
    result = []
    for i in range(duration):
        result.extend(random.sample(range(i * 60, (i + 1) * 60), ntime))
    result.sort()
    return result

def define_patterns(stimulation_timing, ntime, pattern_select):
    """
    stimulation_timing (array): output of generate_stimulation_timing
    duration (int): duration (min)

    output:
    change_time (array): time to change pattern (msec)
    pattern_index (array): which pattern to change
    """
    change_time = []
    pattern_index = []

    if stimulation_timing[0] != 0:
        change_time.append(0)
        pattern_index.append(0)

    for (t, p) in zip(stimulation_timing, pattern_select):
        for i in range(ntime):
            change_time.append(t * 1000 + i * (STIM_DURATION + INTERVAL_DURATION))
            pattern_index.append(p)
            change_time.append(t * 1000 + i * (STIM_DURATION + INTERVAL_DURATION) + STIM_DURATION)
            pattern_index.append(0)

    return change_time, pattern_index


# define the experimental conditions

duration = 15 # min
nStimtime = 60 # number of stim / min
nPulse = 5    # number of pulse / stim


# pattern sequences 
# pattern_select[0] = 1 means the full exposure is chosen in the first session (session means the set of nStimtime burst stimulations)

pattern_select = [2, 3] * (duration * nStimtime // 2) # For full exposure
# pattern_select = [random.sample(range(2, 4), duration * nStimtime)] # for random half exposure

stim_time = generate_stimulation_timing(nStimtime, duration)
cng_t, pindex = define_patterns(stim_time, nPulse, pattern_select)


# prepare a log file

if not os.path.exists(LOGFILR_DIR):
    os.makedirs(LOGFILR_DIR)
now = datetime.datetime.now()
log_file = open(LOGFILR_DIR + '/experiment_' + now.strftime('%Y%m%d_%H%M%S') + '.log', 'x')


# define TCP server address
host = "localhost"
port_stim  = 2222
port_camera  = 2226

# connect to server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
remote_ip = socket.gethostbyname( host )
s.connect((remote_ip, port_stim))

s_camera = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
remote_ip_camera = socket.gethostbyname( host )
s_camera.connect((remote_ip_camera, port_camera))

# define the pattern image size
w = np.uint32(STIM_EDGE); h = np.uint32(STIM_EDGE)

pattern_seq = generate_image_sequence(0)

func = np.uint32(1)
s.send(func)
s.send(w)
s.send(h)
s.send(pattern_seq[1])

time.sleep(1)

# Receive Camera Image
oasis_camera.RequestAllImages(s_camera)
oasisReadImage = oasis_camera.ReadImage(s_camera)
oasis_image = oasisReadImage.receive()
fov_center_x, fov_center_y, _ = detectOuterCircle(oasis_image)

s.send(func)
s.send(w)
s.send(h)
s.send(pattern_seq[0])


half_patt_border_x = w * fov_center_x // oasisReadImage.width

# define stimulation patterns

pattern_seq = generate_image_sequence(half_patt_border_x)

# log start time
log_file.write('start session : ' + str(datetime.datetime.now()) + '\n')
start_time = None

for (cng, p) in zip(cng_t, pindex):

    t = time.time()
    if start_time is None: start_time = t
    while(t - start_time < cng / 1000): t = time.time()
    log_file.write("{:.7f}".format(t - start_time) + ' sec: pattern ' + str(p) + '\n')

    func = np.uint32(2)
    s.send(func)
    s.send(w)
    s.send(h)
    s.send(pattern_seq[p])

# log end time
log_file.write('end session : ' + str(datetime.datetime.now()) + '\n')

# close TCP connection
s.close()
