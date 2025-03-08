# import libraries
import socket
import numpy as np
import time
import os
import sys
import datetime
import random
import cv2


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
    n = width % 8
    if n != 0:
        pad = np.zeros((img.shape[0], 8 - n), dtype=np.uint8)
        img = np.hstack((img, pad))
    return Convert(img).astype(np.uint8)

# stimulation parameters

STIM_DURATION = 40 # msec
INTERVAL_DURATION = 160 # msec

LOGFILR_DIR = './logs'


def generate_image_sequence():
    """
    generate 3 patterns
    0: off
    1: full
    2: left
    3: right
    """

    pattern_seq = []

    # Image 0 (off)
    img = np.zeros((200, 200), dtype=np.uint8) * 255
    pattern_seq.append(FormatImage(img))

    # Image 1 (full exposure)
    img = np.ones((200, 200), dtype=np.uint8) * 255
    pattern_seq.append(FormatImage(img))

    half_black = np.zeros((200, 100), dtype=np.uint8)
    half_white = np.ones((200, 100), dtype=np.uint8) * 255

    # Image 2 (half exposure: left)
    img = np.hstack((half_black, half_white))
    # img = img.transpose()  # if needed
    pattern_seq.append(FormatImage(img))

    # Image 3 (half exposure: left)
    img = np.hstack((half_white, half_black))
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
    return result

def define_patterns(stimulation_timing, ntime, duration, pattern_select):
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
nStimtime = 5 # number of stim / min


# generating patterns

pattern_seq = generate_image_sequence()


# pattern sequences 
# pattern_select[0] = 1 means the full exposure is chosen in the first session (session means the set of nStimtime burst stimulations)

pattern_select = [1] * duration * nStimtime # For full exposure
# pattern_select = [random.sample(range(2, 4), duration * nStimtime)] # for random half exposure

stim_time = generate_stimulation_timing(nStimtime, duration)
cng_t, pindex = define_patterns(stim_time, nStimtime, duration, pattern_select)


# prepare a log file

if not os.path.exists(LOGFILR_DIR):
    os.makedirs(LOGFILR_DIR)
now = datetime.datetime.now()
log_file = open(LOGFILR_DIR + '/experiment_' + now.strftime('%Y%m%d_%H%M%S') + '.log', 'x')


# define TCP server address
host = "localhost"
port = 2222

# connect to server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
remote_ip = socket.gethostbyname( host )
s.connect((remote_ip, port))

# define the pattern image size
w = 200; h = 200

# log start time
log_file.write('start session : ' + str(datetime.datetime.now()) + '\n')
start_time = None

for (cng, p) in zip(cng_t, pindex):

    t = time.time()
    if start_time is None: start_time = t
    while(t - start_time < cng / 1000): t = time.time()
    log_file.write("{:.7f}".format(t - start_time) + ' sec: pattern ' + str(p) + '\n')

    func = np.uint32(1)
    s.send(func)
    s.send(w)
    s.send(h)
    s.send(pattern_seq[p])

# log end time
log_file.write('end session : ' + str(datetime.datetime.now()) + '\n')

# close TCP connection
s.close()
