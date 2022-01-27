import cv2
import requests
import numpy as np
import imutils
import tensorflow as tf
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import glob
from matplotlib import pyplot
import pandas as pd
import os

url = "http://192.168.50.102:8080/shot.jpg"

# While loop to continuously fetching data from the Url
while True:
    
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1080, height=1920)
    copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150) 
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="top-to-bottom")[0]
    
    
    for c in cnts:
      if cv2.contourArea(c) >= 1000:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 10)
    
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img",copy)

    # Press Esc key to exit
    if cv2.waitKey(1) == 27:
      cv2.imwrite('sample.jpg',img)
      break
  
cv2.destroyAllWindows()