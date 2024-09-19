# import cv2
# from picamera2 import Picamera2

# # Initialize OpenCV window
# cv2.startWindowThread()

# # Initialize Picamera2 and configure the camera
# picam2 = Picamera2()
# picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
# picam2.start()

# while True:
#     # Capture frame-by-frame
#     im = picam2.capture_array()

#     # Display the resulting frame
#     cv2.imshow("Camera", im)
   
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close windows
# cv2.destroyAllWindows()







# import cv2
# from picamera2 import Picamera2
# import time

# cv2.startWindowThread()
# # picam2 = picamera2.Picamera2()
# picam2 = Picamera2()
# # picam2.configure(picam2.create_preview_configuration())
# picam2.start()

# # Capture an image array
# time.sleep(2)
# while True:
#     image = picam2.capture_array()
#     cv2.imshow("Camera", image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Do something with the image, e.g., save it
# # ...
# cv2.destroyAllWindows()
# picam2.stop()

import picamera2
import threading
import cv2

def capture_image():
    while True:
        picam2 = picamera2.Picamera2()
        picam2.configure(picam2.create_still_configuration())
        picam2.start()
        image = picam2.capture_array()
        picam2.stop()

        cv2.imshow("Camera", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Do something with the image

thread = threading.Thread(target=capture_image)
thread.start()