# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
import sys
json_list1=[]
imgpath = ""

args = sys.argv

for i in range(1, len(args)):
    if i == 1:
        imgpath = imgpath + args[i]
    else:
        imgpath = imgpath + " " + args[i]

imagepath = os.path.join(imgpath, "images")

#savepath = (".\Validation folder\Annotatedimages\\")
#for image in os.listdir(".\Test folder\images"):
for image in os.listdir(imagepath):
    #img = cv2.imread(".\Test folder\images\\" + image)
    img = cv2.imread(imagepath + "\\" + image)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +  'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        elementdata={"iname":image, "bbox" :[float(x), float(y), float(w), float(h)]}
        json_list1.append(elementdata)
    #cv2.imwrite(image, img)    
output_json = (os.getcwd()+"\\results.json")
with open(output_json, 'w') as f:
    json.dump(json_list1, f)

