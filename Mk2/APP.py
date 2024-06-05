import tkinter as tk
import cv2
from tkinter import *  
from PIL import Image, ImageTk
import numpy as np
import time
import sys
from threading import Thread
from time import sleep


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

model = YOLO("yolov8n.pt")

YOLO_VERBOSE = False

class VideoRectangleDrawerApp:
    def __init__(self, root):
        self.triggered = False
        self.root = root
        self.detectables = []
        self.root.title("Intelligent AI Video Detector")

        self.frame = tk.Frame(self.root)
        self.frame.pack()

        self.detection_canvas = tk.Canvas(self.frame, width=640, height=480)
        self.detection_canvas.pack(side=tk.LEFT)
        # self.drawable_canvas.place(x=0, y=0)

        self.video_canvas = tk.Canvas(self.frame, width=640, height=480)
        self.video_canvas.pack(side=tk.RIGHT)
        # self.video_canvas.place(x=0, y=0)

        self.video_source = "http://192.168.4.1/cam-hi.jpg"  # Change this to your video file
        self.video = cv2.VideoCapture(self.video_source)

        Thread(target = self.update_tracking).start()
        Thread(target = self.update_video).start()
        # thread.start()
        # thread.join()
        # thread1.start()
        # thread1.join()
        # self.update_video()
        # self.update_tracking()

        self.rect_start = None
        self.rect_end = None

        self.video_canvas.bind("<Button-1>", self.on_click)
        self.video_canvas.bind("<B1-Motion>", self.on_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self.on_release)

        

        #Create Buttons and List options

        # self.background_button = tk.Button(self.frame, text="Detect New Item", command=self.detectNewItem)
        # self.background_button.pack()

        # self.title = tk.Text(root, text = "Detection classes:" + str(model.names), bg='lightyellow')
        # # self.title.grid(row=0,column=0)
        # self.title.pack(side=tk.BOTTOM)

        self.title = Text(root, height = 30, width = 100)
        # self.title.grid(row=0,column=0)
        self.title.pack(side=tk.LEFT)
        self.title.insert(tk.END, "Detection classes:" + str(model.names))

        self.display = tk.Label(root, text = "", bg='white')
        # self.display.grid(row=0,column=1)
        self.display.pack(side=tk.BOTTOM)

        self.display1 = tk.Label(root, text = "", bg='white')
        # self.display.grid(row=0,column=1)
        self.display1.pack(side=tk.BOTTOM)

        self.txt_input = tk.Entry(root, width=15)
        # self.txt_input.grid(row=1,column=1)
        self.txt_input.pack(side=tk.RIGHT)

        self.btn_add_task = tk.Button(root, text = "Add Entry", fg = 'black', bg = None, command= Thread(target =self.add_task).start)
        self.btn_add_task.pack(side=tk.RIGHT)
        # self.btn_add_task.grid(row=1,column=0)

        self.btn_delete = tk.Button(root, text = "Delete", fg = 'black', bg = None, command = self.delete)
        self.btn_delete.pack(side=tk.RIGHT)
        # self.btn_delete.grid(row=2,column=0)


        self.lb_tasks = tk.Listbox(root)
        self.lb_tasks.pack(side=tk.RIGHT)
        # self.lb_tasks.grid(row=2,column=1,rowspan=7)


    def update_video(self):
        # ret, frame = self.video.read()
        # if ret:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     frame = cv2.resize(frame, (400, 400))
        #     self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        #     self.video_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        #     self.video_canvas.image = self.photo
        #     self.video_canvas.after(10, self.update_video)
        while(True):
            self.video = cv2.VideoCapture(self.video_source)
            ret, frame = self.video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.video_canvas.create_image(50, 0, image=self.photo, anchor=tk.NW)
                if self.detectables != []:
                    self.video_canvas.create_rectangle(self.rect_start[0], self.rect_start[1],
                                                    self.rect_end[0], self.rect_end[1],
                                                    outline="blue", tags="temp_rect")
                # self.video_canvas.after(500, self.update_video)
                # self.detection_canvas.after(200, self.update_video)


    def update_tracking(self):
        while True:
            self.video = cv2.VideoCapture(self.video_source)
            ret, frame = self.video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                # self.photo1 = ImageTk.PhotoImage(image=Image.fromarray(frame))

                # AI PROCESSING
                # if frame is not None:
                #     # print("Image shape:", frame.shape)
                # else:
                #     print("Failed to decode image.")

                results = model.track(frame, persist=True, verbose=False)
                # print(results[0].boxes)
                if (results[0].boxes.id != None):
                    if (results[0].boxes.id[0]):
                        clss = results[0].boxes.cls.cpu().tolist()
                        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                        ids = results[0].boxes.id.cpu().numpy().astype(int)
                        names = model.names
                        # print(results)
                        for box, cls, id in zip(boxes, clss, ids):
                            # print(box)
                            # print(id)
                            x, y, w, h = box
                            x = x+w/2
                            y = y+h/2
                            for detectable in self.detectables:
                                if cls == detectable[0] and detectable[1][0] < x < detectable[2][0] and detectable[1][1] < y < detectable[2][1]:
                                    # print(model.names.get(cls) + " detected in limit boundaries.")
                                    self.display1['text'] = "OBJECTS DETECTED WITHIN BOUNDARIES: " + model.names.get(cls)
                                else:
                                    self.display1['text'] = "NO OBJECTS DETECTED WITHIN BOUNDARIES" 

                            # print(cls)
                            # print(x)
                            # print(y)
                            # self.isIntersecting(cls, x, y)
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
                            
                self.photo1 = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.detection_canvas.create_image(50, 0, image=self.photo1, anchor=tk.NW)

                # self.detection_canvas.after(1000, self.update_tracking)


    def isIntersecting(self, classOf, xcord, ycord):
        if classOf in self.detectables:
            self.detectables[self.detectables.index(classOf)[0]]


    
    def on_click(self, event):
        self.rect_start = (event.x, event.y)

    def on_drag(self, event):
        if self.rect_start:
            if self.rect_end:
                self.video_canvas.delete("temp_rect")
            self.rect_end = (event.x, event.y)
            self.video_canvas.create_rectangle(self.rect_start[0], self.rect_start[1],
                                                  self.rect_end[0], self.rect_end[1],
                                                  outline="blue", tags="temp_rect")

    def on_release(self, event):
        self.triggered = True
        # if self.rect_start and self.rect_end:
            # print("Rectangle Coordinates:")
            # print("Top Left:", self.rect_start)
            # print("Bottom Right:", self.rect_end)
            # You can store the coordinates in any way you like here
            # self.rect_start = None
            # self.rect_end = None

    # def detectNewItem(self):
    #     # self.video = cv2.VideoCapture(self.video_source)
    #     # ret, frame = self.video.read()
    #     # if ret:
    #     #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     #     frame = cv2.resize(frame, (640, 480))
    #     #     self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    #     #     self.drawable_canvas.create_image(50, 0, image=self.photo, anchor=tk.NW)
    #     #     print("set")
    #     #     # self.video_canvas.after(200, self.update_video)
    #     self.rect_start = None
    #     self.rect_end = None
    #     while True:
    #         if (self.rect_start and self.rect_start != None):


    def update_listbox(self):
        # Clear the current list
        self.clear_listbox()

        #update items to list
        for detectable in self.detectables:
            self.lb_tasks.insert("end", detectable)

    def clear_listbox(self):
        self.lb_tasks.delete(0,"end")


    def add_task(self):
        # Get the task
        task = int(self.txt_input.get())
        # Append the task to list
        if task in model.names:
            self.display['text'] = "Registered. Now drag a detection area."
            self.rect_start = None
            self.rect_end = None
            self.triggered = False
            while True:
                if (self.triggered):
                    self.detectables.append([task, self.rect_start, self.rect_end])
                    self.update_listbox()
                    break
                time.sleep(0.2)
            self.display['text'] = "Done"
        else:
            self.display['text'] = "Please enter a number for a viable object"
        self.txt_input.delete(0,'end')
        # Thread.join()
        sys.exit()

    def delete(self):
        task = self.lb_tasks.get('active')
        if task in self.detectables:
            self.detectables.remove(task)
        # Update list box
        self.update_listbox()

        self.display['text'] = "Task deleted!"

    def delete_all(self):
        # global tasks
        # Clear the list
        self.detectables = []

        self.update_listbox()






if __name__ == "__main__":
    root = tk.Tk()
    # root.configure(bg='')
    app = VideoRectangleDrawerApp(root)
    root.mainloop()