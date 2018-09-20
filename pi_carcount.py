#import packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import warnings
import datetime
import json
import time
import cv2

#argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the JSON configuration file")
args = vars(ap.parse_args())

#filter warnings and load the configuration
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))

#initialize camera and obtain reference image
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]
rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))

#warmup camera, initialize average frame, initialize timestamp, initialize frame motion counter, create data file
print("Warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0
with open("datafile.txt","w") as f:
    f.write("time detected\n")


#begin image stream from camera
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    #obtain raw NumPy array, initialize timestamp
    frame = f.array
    timestamp = datetime.datetime.now()
    detect = False

    #convert image to grayscale to make background comparisons
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    #initialize background frame
    if avg is None:
        print("Starting background model...")
        avg = gray.copy().astype("float")
        rawCapture.truncate(0)
        print("Begin Recording")
        continue

    #determine weighted average between frames, as well as difference between current and average frames
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    #threshold difference image, dilate it, and find contours
    thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]


    #loop over contours
    for c in cnts:
        #ignore small contours
        if cv2.contourArea(c) < conf["min_area"]:
            continue
        detect = True
        
    #Check detection
    if detect == True:
        #check if minimum time has elapsed
        if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
            motionCounter += 1
            #check if movement lasts long enough
            if motionCounter >= conf["min_motion_frames"]:
                #record observation to data text file
                with open("datafile.txt","a") as f:
                    f.write(str(timestamp)+'\n')
                print("Data Recorded")

                #update last uploaded timestamp and reset motion counter
                lastUploaded = timestamp
                motionCounter = 0

    #otherwise, no detection occurs
    else:
        motionCounter = 0

    #clear stream to prep for next frame
    rawCapture.truncate(0)
