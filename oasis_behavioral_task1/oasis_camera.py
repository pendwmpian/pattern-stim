import numpy
import struct

def Receive(s, n):
    rbuf = s.recv(n)
    if len(rbuf) == 0:
        print("recv == 0")
        return rbuf
    while len(rbuf) < n:
        rbuf2 = s.recv(n - len(rbuf))
        if len(rbuf2) == 0:
            print("recv2 == 0")
            return rbuf2
        rbuf = rbuf + rbuf2
    return rbuf

def RequestAllImages(s):
    func = struct.pack('i', 21)
    s.send(func)
    print("Request 21 sent")

def StopAllImages(s):
    func = struct.pack('i', 22)
    s.send(func)
    print("Request 22 sent")

def GetLatestImage(s):
    func = struct.pack('i', 11)
    s.send(func)
    print("Request 11 sent")

class ReadImage():
    def __init__(self, socket_camera):
        self.s_camera = socket_camera
        buf = Receive(self.s_camera, 24)
        if len(buf) != 24:
            raise RuntimeError("Failed to receive header")
        (self.sessionTime, self.width, self.height, self.bitDepth, self.endian, self.pattern) = struct.unpack('qiihhi', buf)
        bps = (self.bitDepth + 7) // 8
        self.pixelSize = self.width * self.height * bps

    def receive(self):
        buf = Receive(self.s_camera, self.pixelSize)
        if len(buf) != self.pixelSize:
            raise RuntimeError("Failed to receive pixel data")
        frame = numpy.frombuffer(buf, dtype=numpy.uint16)
        frame = frame.reshape(self.height, self.width)
        return frame
    
    def height(self):
        return self.height
    def width(self):
        return self.width