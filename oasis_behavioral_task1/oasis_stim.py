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



# stimulation parameters
STIM_DURATION = 10 # msec
INTERVAL_DURATION = 90 # msec
STIM_NUMBER = 20 # times

# session parameters
nSessions = 30 # number of sessions
session_duration = [30, 40] # duration (sec) up to 65535 seconds
# ex. [30, 40] means session durations (sec) are randomly picked up between 30-40sec (which contains 11 patterns 30, 31, ..., 39, 40)

# log files location
LOGFILR_DIR = './logs'

# define arduino address
arduino = serial.Serial(
    port = 'COM3',
    baudrate = 115200,
    parity = serial.PARITY_NONE,
    stopbits = serial.STOPBITS_ONE,
    bytesize = serial.EIGHTBITS)
arduino_lock = threading.Lock() # For thread-safe accessing to arduino I/O

# define TCP server address
host = "localhost"
port = 2222


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
    

def generate_image_sequence():
    """
    generate 4 patterns
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
s.connect((remote_ip, port))

# define the pattern image size
w = 200; h = 200


# stimulation func (Thread1)

def polygon_stimulation(cng_t, pindex, answer, duration, stim_log):

    start_time = None
    with arduino_lock:
        payload = answer.to_bytes(1, 'little')
        payload += duration.to_bytes(2, 'little')
        arduino.write(payload)

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

def task_recording(task_log):

    session_fin = False
    print_cnt = 0

    while(session_fin is False):
        time.sleep(0.001)
        with arduino_lock:
            if arduino.in_waiting > 0:
                str = arduino.readline()

                match str.split(':')[0]:

                    case 'Dist':
                        cnt += 1
                        logging(task_log, str, True if cnt % 10 == 0 else False)

                    case 'Reward':
                        logging(task_log, str, True)

                    case 'Session finished':
                        logging(task_log, str, True)
                        session_fin = True
                
                    case _:
                        logging(task_log, str, True)
                        session_fin = True


log_file_task.write('start sessions : ' + str(datetime.datetime.now()) + '\n\n')


for session in range(nSessions):

    duration = random.sample(range(session_duration[0], session_duration[1] + 1), 1)[0] # session duration (sec)
    stim_LR = random.sample(range(2, 4), 1)[0] # Left(2) or Right(3) to stimulate 
    answer = stim_LR # Correct position to stay in the linear track: Left(2) or Right(3)

    cng_t, pindex = define_one_patterns(stim_LR)

    thread_1 = threading.Thread(target=polygon_stimulation, args=(cng_t, pindex, answer, duration, log_file_stim,))
    thread_2 = threading.Thread(target=task_recording, args=(log_file_task,))

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
arduino.close()

# close log files
log_file_stim.close()
log_file_task.close()