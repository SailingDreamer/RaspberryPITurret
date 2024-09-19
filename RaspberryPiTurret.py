import math
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
from threading import Thread
from gpiozero import AngularServo

# from picamera.array import PiRGBArray
from picamera2 import Picamera2
import time
# import RPi.GPIO as GPIO
import cv2

# GPIO.setmode(GPIO.BOARD)

servoYawPin = 13
servoPitchPin = 16
servoTriggerPin = 19

# GPIO.setup(servoYawPin, GPIO.OUT)
# GPIO.setup(servoPitchPin, GPIO.OUT)
# GPIO.setup(servoTriggerPin, GPIO.OUT)

# pwm=GPIO.PWM(servoYawPin, 50)
# pwm=GPIO.PWM(servoPitchPin, 50)
# pwm=GPIO.PWM(servoTriggerPin, 50)

# pwm.start(0)

servoYaw = AngularServo(servoYawPin, min_pulse_width=0.0005, max_pulse_width=0.0025)
servoPitch = AngularServo(servoPitchPin, min_pulse_width=0.0005, max_pulse_width=0.0025)
servoTrigger = AngularServo(servoTriggerPin, min_pulse_width=0.0005, max_pulse_width=0.0025)

servoYaw = 0
servoPitch = 0
servoTrigger = 0

forwardTriggerServoLimit = 0
backwardTriggerServoLimit = 50
fireingIsActive = False


model = YOLO('yolov8l.pt')

# waitTime = 0.2
waitTime = 0.0

numCyclesWaitUntillAut = 20
cyclesSinceLastHumanDetected = numCyclesWaitUntillAut

autoRotateTurretManagerDirection = True

# print(model)
t0 = 0
t1 = 0

brightness = 8
contrast = 2.1

degreeRotatedX = 70
degreeRotatedYNormal = 55

# camera = PiCamera()
# rawCapture = PiRGBArray(camera)

# Initialize OpenCV window
cv2.startWindowThread()


# Initialize Picamera2 and configure the camera

# picam2 = Picamera2()
# picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
# picam2.start()

picam2 = Picamera2()
picam2.stop()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

im=None

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

def setAngle(angle, servoPin):
	# duty = angle / 18 + 2
	# GPIO.output(servoPin, True)
	# pwm.ChangeDutyCycle(duty)
    servoPin.angle = angle
	# GPIO.output(servoPin, False)
	# pwm.ChangeDutyCycle(0)
    
def fireWeapon(forwardLimit, backwardsLimit):
    fireingIsActive = True
    setAngle(forwardLimit, servoTrigger)
    time.sleep(1)
    setAngle(backwardsLimit, servoTrigger)
    time.sleep(1)
    fireingIsActive = False

def fireWeaponThread():
    fireWeapon(forwardTriggerServoLimit, backwardTriggerServoLimit)

def centerCordinates(width, height, cordinateX, cordinateY):
    return cordinateX -(width/2), cordinateY - (height/2)

def rotateSignal(degreeRotDeltax, degreeRotDeltay, timedifference, fire):
    global t0
    if time.time() - t0 >= timedifference:
        # print("Time since calc" + str(time.time()-t1))

        global degreeRotatedX
        degreeRotatedX -= int(degreeRotDeltax)

        
        global degreeRotatedYNormal
        degreeRotatedY = degreeRotDeltay + degreeRotatedYNormal
        

        setAngle(degreeRotatedX, servoYaw)
        setAngle(degreeRotatedY, servoPitch)

        Thread(target = fireWeaponThread).start()
        # SetAngle(servoTriggerPin)

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

# formatAndSend(degreeRotatedX,degreeRotatedY,0)

while True:

    # img_resp=urllib.request.urlopen(url)
    # imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    # im = cv2.imdecode(imgnp,-1)
    # im = cv2.addWeighted(im, contrast, np.zeros(im.shape, im.dtype), 0, brightness)

    # time.sleep(0.1)



    # # grab an image from the camera
    # camera.capture(rawCapture, format="bgr")
    # im = rawCapture.array
    # # display the image on screen and wait for a keypress
    # cv2.imshow("Raw Image", im)

    time.sleep(waitTime)

    # Capture frame-by-frame
    im = picam2.capture_array()


    if im is None:
        # print("Image shape:", im.shape)
        print("Failed to decode image.")
        continue


    # bbox, label, conf = cv.detect_common_objects(im)
    # print(bbox)
    # if(im):
    # if im is not None:
    #     print("Image shape:", im.shape)
    # else:
    #     print("Failed to decode image.")
    # print(im.shape)

    print("start")

    cv2.imshow('Camera Input',im)

    # global model
    print("aksjfd;akls;gkfjkg;l")
    print(model)
    results = model.track(im, persist=False)
    # results = model.track(source=im, persist=True, tracker="bytetrack.yaml", verbose=False)

    print("end")
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

                    #     t0 = time.time()fireWeaponThread

    #print(label)
    # if bbox and label == "person" :
    #     print(bbox[0]-(bbox[2]/2))#finding centered box point
    #     print(bbox[1]-(bbox[3]/2))
    
    # print(bbox)

    if numCyclesWaitUntillAut <= cyclesSinceLastHumanDetected:
        autoRotateTurretManager(2, currY, 40, 120, 0.5)

    cyclesSinceLastHumanDetected += 1

    cv2.imshow('Detection',im)

    key=cv2.waitKey(5)
    if key==ord('q'):
        break

    print("cycled")

picam2.stop()
servoYaw.close()
servoPitch.close()
servoTrigger.close()

# pwm.stop()
# GPIO.cleanup()

cv2.destroyAllWindows()
 
 
# while(True):
#     run2()
# if __name__ == '__main__':
#     print("started")
#     with concurrent.futures.ProcessPoolExecutor() as executer:
#         # f1= executer.submit(run1)
#         f2= executer.submit(run2)