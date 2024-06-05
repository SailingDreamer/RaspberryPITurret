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
print("connecting to socket")
s = socket.socket()        
host = '192.168.4.1' 
port = 79   
# model = YOLO('yolov5nu.pt')
im = None
model = YOLO('yolov8l.pt')

waitTime = 0.2
numCyclesWaitUntillAut = 20

autoRotateTurretManagerDirection = True
# print(model)
t0 = 0
t1 = 0

brightness = 8
contrast = 2.1

degreeRotatedX = 70
degreeRotatedY = 55
currY = degreeRotatedY


def formatAndSend(yaw, pitch, trig):
    DTCPMessage = zeroize(yaw) + zeroize(pitch) + str(trig) + 'a'
    s.sendall(DTCPMessage.encode())

def zeroize(val): 
    numOfZeros = 3-len(str(val))
    return numOfZeros*'0' + str(val)


s.connect((host, port))

#camera socket
url='http://192.168.4.1/cam-hi.jpg'
im=None
cyclesSinceLastHumanDetected = numCyclesWaitUntillAut
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

def centerCordinates(width, height, cordinateX, cordinateY):
    return cordinateX -(width/2), cordinateY - (height/2)

def rotateSignal(degreeRotDeltax, degreeRotDeltay, timedifference, fire):
    global t0
    if time.time() - t0 >= timedifference:
        # print("Time since calc" + str(time.time()-t1))

        global degreeRotatedX
        degreeRotatedX -= int(degreeRotDeltax) #adding change to current rotation value

        #degreeRotatedY += int(degreeRotDeltay) camera only rotates with x axis not y axis
        
        global degreeRotatedY
        global currY 
        currY = degreeRotDeltay + degreeRotatedY
        formatAndSend(int(degreeRotatedX),int(currY),fire)
        print("data sent")
        t0 = time.time()

def autoRotateTurretManager(amount, currY, xLimLow, xLimHigh, timedifference):
    global t1
    if time.time() - t1 >= timedifference:
        global degreeRotatedX
        # global degreeRotatedY
        global autoRotateTurretManagerDirection
        # print(degreeRotatedX)
        if ((degreeRotatedX - amount <= xLimLow) or (degreeRotatedX + amount >= xLimHigh)):

            # autoRotateTurretManagerDirection = not autoRotateTurretManagerDirection

            if (degreeRotatedX - amount <= xLimLow):
                autoRotateTurretManagerDirection = False
                #initial offset
                rotateSignal(-amount-1, 0, 0.1, 0)
            elif (degreeRotatedX + amount >= xLimHigh):
                autoRotateTurretManagerDirection = True   
                #initial offset
                rotateSignal(amount + 1, 0, 0.1, 0)
        else:
            if autoRotateTurretManagerDirection:
                # degreeRotatedX -= amount
                # rotateSignal(degreeRotatedX, degreeRotatedY, 0.5)
                rotateSignal(amount, 0, 0.1, 0)
            else:
                # degreeRotatedX += amount
                # rotateSignal(degreeRotatedX, degreeRotatedY, 0.5)
                rotateSignal(-amount, 0, 0.1, 0)
        t1 = time.time()

        
# def run2():


# degreeRotatedY = 99
#formatAndSend(degreeRotatedX,degreeRotatedY,1)
# cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)



# def periodicRequest():
#     while True:
#         img_resp=urllib.request.urlopen(url)
#         imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
#         ima = cv2.imdecode(imgnp,-1)
#         if ima is None:
#             # print("Image shape:", im.shape)
#             print("Failed to decode image.")
#         else:
#             global im 
#             im = ima
#             print("received")
#             time.sleep(waitTime)

# Thread(target = periodicRequest).start()

formatAndSend(degreeRotatedX,degreeRotatedY,0)

while True:

    cyclesSinceLastHumanDetected += 1

    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    im = cv2.imdecode(imgnp,-1)
    im = cv2.addWeighted(im, contrast, np.zeros(im.shape, im.dtype), 0, brightness)

    if im is None:
        # print("Image shape:", im.shape)
        print("Failed to decode image.")


    # bbox, label, conf = cv.detect_common_objects(im)
    # print(bbox)
    # if(im):
    # if im is not None:
    #     print("Image shape:", im.shape)
    # else:
    #     print("Failed to decode image.")
    # print(im.shape)

    # print("start")

    # results = model.track(im, persist=True)
    results = model.track(source=im, persist=True, tracker="bytetrack.yaml", verbose=False)

    # print("stop")
    # print("Results")
    # print(results[0].boxes.id)


    # if results[0] != None:

    #     boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    #     ids = results[0].boxes.id.cpu().numpy().astype(int)
    #     print("length of the results" + results.length)
    # t1 = time.time()

    if results[0].boxes.id != None:
        if (results[0].boxes.id[0]):

            # Extract prediction results
            # clss = results[0].boxes.cls.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.int().cpu().tolist()
            # confs = results[0].boxes.conf.float().cpu().tolist()
            # print()
            # print(boxes)
            # print()
            # print(ids)
        
            names = model.names
            for box, cls, id in zip(boxes, clss, ids):
                # print(box)
                im = cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
                cv2.putText(
                im,
                f"Id {id}",
                (box[0], box[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                )
                # print("here ")
                
                if cls == 0:
                    # xcor = bbox[index][0]+(bbox[index][2]/2)#finding centered box point
                    # ycor = bbox[index][1]+(bbox[index][3]/2)
                    # print(xcor)
                    # print(ycor)
                    #im = draw_bbox(im, [[int(bbox[index][0]), int(bbox[index][1]), int(bbox[index][0]) + 1, int(bbox[index][0] + 1)]], label, conf)#[[int(xcor), int(ycor),1 + int(xcor),1 + int(ycor)]]
                    # print(box[0])


                    # cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    # cv2.putText(
                    # im,
                    # f"Id {id}",
                    # (box[0], box[1]),
                    # cv2.FONT_HERSHEY_SIMPLEX,
                    # 1,
                    # (0, 0, 255),
                    # 2,
                    # )


                    # centeredCordx = int(box[0]) + (int(box[0]) - int(box[2]))/2 #getting centered cooridnates
                    # centeredCordy = int(bbox[index][1]) + (int(bbox[index][3]) - int(bbox[index][1]))/5

                    centeredCordx = int(box[0]) - (int(box[0]) - int(box[2]))/2 #getting centered cooridnates
                    centeredCordy = int(int(box[1]) - (int(box[1]) - int(box[3]))/5)
                    
                    #print(centeredCordx)
                    #print(centeredCordy)


                    im = cv2.rectangle(im, (int(centeredCordx), int(centeredCordy)), (10 + int(centeredCordx),10 + int(centeredCordy)), (255, 225, 0) , 2)


                    h, w, c = im.shape

                    centeredCordx = centeredCordx - w/2# normalizing cordinates so the center of the screen is 0,0
                    centeredCordy = centeredCordy - h/2
                    degreeRotDeltax = int(math.atan(centeredCordx/800) * (180 / math.pi))
                    degreeRotDeltay = int(math.atan(centeredCordy/800) * (180 / math.pi))
                    
                    # print(degreeRotDeltax)
                    # print(degreeRotDeltay)
                    # print()


                    # print(degreeRotatedX)
                    # print(degreeRotatedY)          

                    # Pitch - 99-50
                    # Yaw - 0 - 180
                    # if(0<degreeRotatedX<180 and 50<degreeRotatedY<99):

                    rotateSignal(degreeRotDeltax, degreeRotDeltay, 1, 1)
                    cyclesSinceLastHumanDetected = 0



                    # if time.time() - t0 >= 0.5:
                    #     # print("Time since calc" + str(time.time()-t1))

                    #     degreeRotatedX -= int(degreeRotDeltax) #adding change to current rotation value

                    #     #degreeRotatedY += int(degreeRotDeltay) camera only rotates with x axis not y axis

                    #     degreeRotatedY = -degreeRotDeltay

                    #     formatAndSend(int(degreeRotatedX),int(degreeRotatedY),1)
                    #     print("data sent                    ----------------------------------------------------------------------")

                    #     t0 = time.time()


                    

                    # cv2.getWindowImageRect('Frame')
                    # im = draw_bbox(im, box, names, 1)


    #print(label)
    # if bbox and label == "person" :
    #     print(bbox[0]-(bbox[2]/2))#finding centered box point
    #     print(bbox[1]-(bbox[3]/2))
    
    # print(bbox)

    if numCyclesWaitUntillAut <= cyclesSinceLastHumanDetected:
        autoRotateTurretManager(2, currY, 40, 120, 0.5)

    cv2.imshow('detection',im)


    # time.sleep(waitTime)



    key=cv2.waitKey(5)
    if key==ord('q'):
        break

    print("cycled")

cv2.destroyAllWindows()
 
 
# while(True):
#     run2()
# if __name__ == '__main__':
#     print("started")
#     with concurrent.futures.ProcessPoolExecutor() as executer:
#         # f1= executer.submit(run1)
#         f2= executer.submit(run2)