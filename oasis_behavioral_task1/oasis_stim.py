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


# stimulation parameters
STIM_DURATION = 10 # msec
INTERVAL_DURATION = 90 # msec
STIM_NUMBER = 20 # times

# session parameters
nSessions = 30 # number of sessions
session_duration = [30, 40] # duration (sec) up to 65024 seconds
# ex. [30, 40] means session durations (sec) are randomly picked up between 30-40sec (which contains 11 patterns 30, 31, ..., 39, 40)

# Define Reward Regions in mm(milli-meters)
LRegionLeftEnd = 100
LRegionRightEnd = 900
RRegionLeftEnd = 1100
RRegionRightEnd = 1900

# log files location
LOGFILR_DIR = './logs'

# define arduino address
arduino_left = serial.Serial(
    port = 'COM3',
    baudrate = 115200,
    parity = serial.PARITY_NONE,
    stopbits = serial.STOPBITS_ONE,
    bytesize = serial.EIGHTBITS)
arduino_right = serial.Serial(
    port = 'COM4',
    baudrate = 115200,
    parity = serial.PARITY_NONE,
    stopbits = serial.STOPBITS_ONE,
    bytesize = serial.EIGHTBITS)
arduino_lock = threading.Lock() # For thread-safe accessing to arduino I/O

# define TCP server address
host = "localhost"
port_stim = 2222
port_camera = 2226

# camera parameter 
FOV_RADIUS = 414
STIM_EDGE = 500

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

    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)

    edges = cv2.Canny(blurred, threshold1=20, threshold2=30)

    circle_template = np.zeros((radius * 2, radius * 2), dtype=np.uint8)
    cv2.circle(circle_template, (radius, radius), radius, 255, 2)

    result = cv2.matchTemplate(edges, circle_template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    center_x, center_y = max_loc[0] + radius, max_loc[1] + radius

    output_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
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



def define_one_patterns(pattern_select, offset = 0):
    """
    pattern_select (int): 2 for left, 3 for right
    offset (int): Time (msec) between the beginning of a task session and the start of stimulation

    output:
    change_time (array): time to change pattern (msec)
    pattern_index (array): which pattern to change
    """
    change_time = []
    pattern_index = []

    if offset > 0:
        change_time.append(0)
        pattern_index.append(0)

    for i in range(STIM_NUMBER):
        change_time.append(offset + i * (STIM_DURATION + INTERVAL_DURATION))
        pattern_index.append(pattern_select)
        change_time.append(offset + i * (STIM_DURATION + INTERVAL_DURATION) + STIM_DURATION)
        pattern_index.append(0)

    return change_time, pattern_index


def logging(log_file, msg, console=False):
    log_file.write(msg)
    if console: print(msg.split('\n')[0])



# prepare a log file

if not os.path.exists(LOGFILR_DIR):
    os.makedirs(LOGFILR_DIR)
now = datetime.datetime.now()
log_file_stim = open(LOGFILR_DIR + '/experiment_' + now.strftime('%Y%m%d_%H%M%S') + '_stimulation.log', 'x')
log_file_task = open(LOGFILR_DIR + '/experiment_' + now.strftime('%Y%m%d_%H%M%S') + '_tasks.log', 'x')



# connect to server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
remote_ip = socket.gethostbyname( host )
s.connect((remote_ip, port_stim))

# s_camera = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# remote_ip_camera = socket.gethostbyname( host )
# s_camera.connect((remote_ip_camera, port_camera))

# # Receive Camera Image
# oasis_camera.RequestAllImages(s_camera)
# oasisReadImage = oasis_camera.ReadImage(s_camera)
# oasis_image = oasisReadImage.receive()
# fov_center_x, fov_center_y, _ = detectOuterCircle(oasis_image)


fov_center_x = 600
# define the pattern image size
w = STIM_EDGE; h = STIM_EDGE
half_patt_border_x = w * fov_center_x // 1280 #oasisReadImage.width

# define stimulation patterns

pattern_seq = generate_image_sequence(half_patt_border_x)


# stimulation func (Thread1)

def polygon_stimulation(cng_t, pindex, answer, duration, stim_log):

    start_time = None
    dur1 = duration % 255; dur2 = duration // 255
    with arduino_lock:
        payload = answer.to_bytes(1, 'little')
        payload += dur1.to_bytes(1, 'little')
        payload += dur2.to_bytes(1, 'little')
        payload += b"\x00"
        arduino_left.write(payload)
        arduino_right.write(payload)

    stim_log.write('start signal to arduino : ' + str(datetime.datetime.now()) + '\n')

    print("### stimulation start ###")

    for (cng, p) in zip(cng_t, pindex):

        t = time.time()
        if start_time is None: start_time = t
        time.sleep(cng / 1000000 - t / 1000 - 0.001)
        while(t - start_time < cng / 1000): 
            t = time.time()
        stim_log.write("{:.7f}".format(t - start_time) + ' sec: pattern ' + str(p) + '\n')

        func = np.uint32(1)
        s.send(func)
        s.send(w)
        s.send(h)
        s.send(pattern_seq[p])

    print("### stimulation end ###")


# task recording func (Thread2)

def task_recording(task_log, answer):

    session_fin = False
    print_cnt = 0

    while(session_fin is False):
        time.sleep(0.05)
        with arduino_lock:
            if arduino_left.in_waiting > 0:
                str = arduino_left.readline()
                distance = [-1000] * 2

                match str.split(':')[0]:

                    case 'Dist(Left)':
                        cnt += 1
                        logging(task_log, str, True if cnt % 10 == 0 else False)
                        distance[0] = int(str.split(' ')[1])

                    case 'Reward':
                        logging(task_log, str, True)

                    case 'Session finished':
                        logging(task_log, str, True)
                        session_fin = True
                
                    case _:
                        logging(task_log, str, True)
                        session_fin = True

                str = arduino_right.readline()

                match str.split(':')[0]:

                    case 'Dist(Right)':
                        cnt += 1
                        logging(task_log, str, True if cnt % 10 == 0 else False)
                        distance[1] = int(str.split(' ')[1])
                
                    case _:
                        pass
                
                if answer == 2:
                    if LRegionLeftEnd < distance[0] and distance[0] < LRegionRightEnd and LRegionLeftEnd < distance[1] and distance[1] < LRegionRightEnd:
                        payload = b"\x40\x00\x00\x00"
                        arduino_left.write(payload)
                if answer == 3:
                    if RRegionLeftEnd < distance[0] and distance[0] < RRegionRightEnd and RRegionLeftEnd < distance[1] and distance[1] < RRegionRightEnd:
                        payload = b"\x40\x00\x00\x00"
                        arduino_left.write(payload)


log_file_task.write('start sessions : ' + str(datetime.datetime.now()) + '\n\n')


for session in range(nSessions):

    duration = random.sample(range(session_duration[0], session_duration[1] + 1), 1)[0] # session duration (sec)
    stim_LR = random.sample(range(2, 4), 1)[0] # Left(2) or Right(3) to stimulate 
    answer = stim_LR # Correct position to stay in the linear track: Left(2) or Right(3)

    cng_t, pindex = define_one_patterns(stim_LR)

    thread_1 = threading.Thread(target=polygon_stimulation, args=(cng_t, pindex, answer, duration, log_file_stim,))
    thread_2 = threading.Thread(target=task_recording, args=(log_file_task, answer,))

    # Session information
    logging(log_file_task, 'Session ' + str(session) + ' : ' + '\n', True)
    logging(log_file_task, 'Duration : ' + str(duration) + ' sec\n', True)
    logging(log_file_task, 'Stimulation : ' + str(stim_LR) + ('. Left' if stim_LR == 2 else '. Right') + '\n', True)
    logging(log_file_stim, 'Session ' + str(session) + '\n')

    thread_1.start()
    thread_2.start()

    # start time
    logging(log_file_task, 'Session Start: ' + str(datetime.datetime.now()) + '\n', True)

    thread_1.join()
    thread_2.join()

    # end time
    logging(log_file_task, 'Session End: ' + str(datetime.datetime.now()) + '\n\n', True)


log_file_task.write('end sessions : ' + str(datetime.datetime.now()) + '\n')


# close TCP connection
s.close()

# close arduino connection
arduino_left.close()
arduino_right.close()

# close log files
log_file_stim.close()
log_file_task.close()