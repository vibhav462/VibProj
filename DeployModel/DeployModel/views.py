from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import cv2
import pandas as pd
import os
import numpy as np
import mahotas as mt
import pickle
import joblib

def home(request):
    return render(request,'home.html',{'name':'home'})

def result(request):
    return render(request,'result.html')

def about (request):
        return render(request,'about.html')
        
def contact (request):
    return render(request,'contact.html')

#def func(result):
    if result == "0":
        return 'zero'
    elif result == "1":
        return 'one'
    elif result == "2":
        return 'two'

def upload(request):
    lower = np.array([10,0,10])
    upper = np.array([100,255,255])
    context = {}
    global result
    global sus
    vector1 = [ ]
    join = ""
    #path = 'C:/Users/Vibhav/Desktop/project/DeployModel'
    if request.method == 'POST':
        upload_file = request.FILES['myfile']
        fs = FileSystemStorage()
        name = fs.save(upload_file.name,upload_file)
        location = fs.url(name)
        print(location)
        #context['url'] = location
        #join = path + location
        for i in range(len(location)):
            if i != 0:
                join = join + location[i]
        #print(join)
        img = cv2.imread(join)
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2HSV)
        
        mask = cv2.inRange(img_hsv, lower, upper)
        img_seg = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        
        gs = cv2.cvtColor(img_seg,cv2.COLOR_RGB2GRAY)
        img_hsv2=cv2.cvtColor(img_seg,cv2.COLOR_RGB2HSV)
        
        feature = cv2.HuMoments(cv2.moments(gs)).flatten()
        haralick = mt.features.haralick(gs).mean(axis=0)
        hist  = cv2.calcHist([img_hsv2], [0, 1, 2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist2 = hist.flatten()
        vector1 = np.hstack([hist2,haralick,feature])
        #print(join) 
        #print(vector1)      
        model = joblib.load('finalised_model1.sav')
        result = model.predict([vector1])
        context = {"Diseased1":"0","Healthy":"1","Diseased2":"2","result":result}
        #sus = func(result)
        #print(type(sus))
        #print(type(result))
        #print(result)
        #print(context)
    return render(request,'upload.html',context)