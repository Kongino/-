import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque
from sklearn.preprocessing import PolynomialFeatures

Fps=30

#https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ko

def detect(addr):
  predic = np.loadtxt(addr, delimiter=',', dtype=np.float32)
  predictions = model(predic)
  count=0
  befo=0
  chcount=0
  max_out=0
  formax_out=0
  max_in=0
  formax_in=0
  more_2sec=0
  for2sec=0
  more_5sec=0
  for5sec=0
  frame_no=0
  que = deque()
  cheat = 0

  for i, logits in enumerate(predictions) :
      if (cheat == 1):
          break
      class_idx = tf.argmax(logits).numpy()
      if (frame_no < Fps*120):
          que.append(class_idx)
      else :
          que.popleft()
          que.append(class_idx)
      frame_no=frame_no+1
      if (frame_no >= Fps*120):
        count=0
        befo=0
        chcount=0
        max_out=0
        formax_out=0
        max_in=0
        formax_in=0
        more_2sec=0
        for2sec=0
        more_5sec=0
        for5sec=0
        frame_no=0
        for j in que:
            if (j==1):
                formax_in=0
                count=count+1
                if (befo == 0):
                    chcount=chcount+1
                else :
                    formax_out=formax_out+1
                    for2sec=for2sec+1
                    for5sec=for5sec+1
                befo=1
            else :
                befo=0
                formax_out=0
                for2sec=0
                formax_in=formax_in+1
            if (formax_out > max_out):
                max_out=formax_out
            if (formax_in > max_in):
                max_in=formax_in
            if (for2sec > Fps*2):
                more_2sec=more_2sec+1
                for2sec=0
            if (for5sec>Fps*5):
                more_5sec=more_5sec+1
                for5sec=0
        
        detect_data = tf.convert_to_tensor([[count, chcount, max_out, max_in, more_2sec, more_5sec]])
        
        poly_ftr=PolynomialFeatures(degree=2).fit_transform(detect_data)
        detection = detect_model(poly_ftr)
        
        #detection=detect_model(detect_data)
        for k, logi in enumerate(detection):
            if (tf.argmax(logi).numpy() == 1):
                print(addr+' - suspect!')
                cheat=1
                break
  if (cheat == 0):
      print(addr+' - Normal')
 




model = tf.keras.models.load_model('C:/Users/inoh9/Desktop/ML/First_model', compile=False)
detect_model = tf.keras.models.load_model('C:/Users/inoh9/Desktop/ML/Second_model', compile=False)

#model.summary()

addr1 = 'C:/Users/inoh9/Desktop/ML/ForT/Abnormal1.csv'
detect(addr1)

addr2 = 'C:/Users/inoh9/Desktop/ML/ForT/Abnormal24.csv'
detect(addr2)

addr3 = 'C:/Users/inoh9/Desktop/ML/ForT/Normal24.csv'
detect(addr3)
