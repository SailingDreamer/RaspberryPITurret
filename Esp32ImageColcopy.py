import math
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
import socket          
import time    
from threading import Thread
#data socket

YOLO_VERBOSE = False

class TurretComputerController:

    def __init__(self):
        print("connecting to socket")
        self.s = socket.socket()        
        host = '192.168.4.1' 
        port = 79   
        # self.model = YOLO('yolov8n.pt')
        self.model = YOLO('yolov8n.pt')
        # print(model)
        self.results = None
        self.degreeRotatedX = 70
        self.degreeRotatedY = 40
        self.s.connect((host, port))

        #camera socket
        self.url='http://192.168.4.1/cam-hi.jpg'
        self.im=None
        Thread(target = self.processPictureThread).start()
        Thread(target = self.processCalculations).start()
        Thread(target = self.processYoloAlg).start()
        Thread(target = self.sendPeriodic).start()
        #formatAndSend(degreeRotatedX,degreeRotatedY,1)
        # cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)

        while True:

            # Pitch - 99-50
            # Yaw - 0 - 180
            
            if self.im is not None:
                print("displayed")
                # if self.results is not None:
                #     print(self.results)

                cv2.imshow('Detection',self.im)
            time.sleep(0.2)
            key=cv2.waitKey(5)
            if key==ord('q'):
                break
                
        cv2.destroyAllWindows()


    def formatAndSend(self, yaw, pitch, trig):
        DTCPMessage = self.zeroize(yaw) + self.zeroize(pitch) + str(trig) + 'a'
        self.s.sendall(DTCPMessage.encode())

    def zeroize(self, val): 
        numOfZeros = 3-len(str(val))
        return numOfZeros*'0' + str(val)
    
    # def run1():
    #     cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
    #     while True:
    #         img_resp=urllib.request.urlopen(url)
    #         imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    #         im = cv2.imdecode(imgnp,-1)
    #         cv2.imshow('live transmission',im)
    #         key=cv2.waitKey(5)
    #         if key==ord('q'):
    #             break
                
    #     cv2.destroyAllWindows()

    def centerCordinates(self, width, height, cordinateX, cordinateY):
        return cordinateX -(width/2), cordinateY -(height/2)

    def processPictureThread(self):
        while True:
            img_resp=urllib.request.urlopen(self.url)
            imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
            self.im = cv2.imdecode(imgnp,-1)
            # print("Image")
            # if self.im is not None:
            #     # print("Image shape:", self.im.shape)
            # else:
            #     print("Failed to decode image.")
            # if self.im is None:
            #     print("Failed to decode image.")

            # print("recieved1")
            # # self.results = self.model.track(self.im, persist=True, verbose=False)
            # self.results = self.model.track(self.im, persist=True, verbose=False)
            # print("recieved")
            time.sleep(0.3)

    def processYoloAlg(self):
        while True:
            if self.im is None:
                print("Failed to decode image.")
            else:
                print("start")
                # self.results = self.model.track(self.im, persist=True, verbose=False)
                self.results = self.model.track(source=self.im, persist=True, tracker="bytetrack.yaml", verbose=True)
                print("stop")
            time.sleep(0.4)
            
    def sendPeriodic(self):
        while True:
            self.formatAndSend(int(self.degreeRotatedX),int(self.degreeRotatedY),1)
            time.sleep(1)

    def processCalculations(self):
        while True:
            
            # print("results")
            # print(self.results)
            if self.results is not None:
                if self.results[-1].boxes.id != None:
                    if self.results[-1].boxes.id[0]:
                        print("here")
                        clss = self.results[-1].boxes.cls.cpu().tolist()
                        boxes = self.results[-1].boxes.xyxy.cpu().numpy().astype(int)
                        ids = self.results[-1].boxes.id.int().cpu().tolist()
                        
                        time.sleep(0.2)

                        for box, cls, id in zip(boxes, clss, ids):

                            self.im = cv2.rectangle(self.im, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
                            cv2.putText(
                            self.im,
                            f"Id {id}",
                            (box[0], box[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                            )
                            # print("here ")
                            
                            if cls == 0:
                                

                                centeredCordx = int(box[0]) - (int(box[0]) - int(box[2]))/2 #getting centered cooridnates
                                centeredCordy = int(int(box[1]) - (int(box[1]) - int(box[3]))/5)
                                
                                #print(centeredCordx)
                                #print(centeredCordy)
                                self.im = cv2.rectangle(self.im, (int(centeredCordx), int(centeredCordy)), (10 + int(centeredCordx),10 + int(centeredCordy)), (255, 225, 0) , 2) 
                                h, w, c = self.im.shape

                                centeredCordx = centeredCordx - w/2# normalizing cordinates so the center of the screen is 0,0
                                centeredCordy = centeredCordy - h/2
                                degreeRotDeltax = int(math.atan(centeredCordx/800) * (180 / 3.141))
                                degreeRotDeltay = int(math.atan(centeredCordy/800) * (180 / 3.141))
                                
                                # print(degreeRotDeltax)
                                # print(degreeRotDeltay)
                                # print()

                                self.degreeRotatedX -= int(degreeRotDeltax) #adding change to current rotation value

                                #degreeRotatedY += int(degreeRotDeltay) camera only rotates with x axis not y axis

                                self.degreeRotatedY = degreeRotDeltay
                                print("calc person                                     OMJFGKJHFglkhflgkjhsdlkfjghslkjfgh")
                                time.sleep(0.1)
                                

    
 
if __name__ == "__main__":
    # root.configure(bg='')
    TurretComputerController()


# while(True):
#     run2()
# if __name__ == '__main__':
#     print("started")
#     with concurrent.futures.ProcessPoolExecutor() as executer:
#         # f1= executer.submit(run1)
#         f2= executer.submit(run2)