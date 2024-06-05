import cv2
from ultralytics import YOLO
import math
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
import socket          
import time    

cap = cv2.VideoCapture(0)
model = YOLO("yolov8n.pt")
url='http://192.168.4.1/cam-mid.jpg'


while True:


    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    frame = cv2.imdecode(imgnp,-1)

    # bbox, label, conf = cv.detect_common_objects(im)
    print("here")
    # print(bbox)
    # if(im):
    if frame is not None:
        print("Image shape:", frame.shape)
    else:
        print("Failed to decode image.")


    # ret, frame = cap.read()


    # if not ret:
    #     break
    results = model.track(frame, persist=True)
    print(results[0].boxes)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    names = model.names
    print("here are the results")
    # print(results)
    for box, id in zip(boxes, ids):
        # print(box)
        # print(id)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Id {id}",
            (box[0], box[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        print(results[0].plot())
        # for c in results.boxes.cls:
        #     print(names[int(c)])
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break