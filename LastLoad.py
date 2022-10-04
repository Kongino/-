import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque


#https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ko


model = tf.keras.models.load_model('C:/Users/inoh9/Desktop/ML/model_1_21_23_test96')
detect_model = tf.keras.models.load_model('C:/Users/inoh9/Desktop/ML/model_detect97')

model.summary()

def detect(predic):
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
  frame_no=0
  que = deque()
  cheat = 0
  suspect_count=0

  for i, logits in enumerate(predictions) :
      if (cheat == 1):
          break
      class_idx = tf.argmax(logits).numpy()
      if (frame_no < 3600):
          que.append(class_idx)
      else :
          que.popleft()
          que.append(class_idx)
      frame_no=frame_no+1
      if (frame_no >= 3600):
        count=0
        befo=0
        chcount=0
        max_out=0
        formax_out=0
        max_in=0
        formax_in=0
        more_2sec=0
        for2sec=0
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
            if (for2sec > 60):
                more_2sec=more_2sec+1
                for2sec=0
        detect_data = tf.convert_to_tensor([[count, chcount, max_out, max_in, more_2sec]])
        detection = detect_model(detect_data)
        for k, logi in enumerate(detection):
            if (tf.argmax(logi).numpy() == 1):
                print('- suspect!')
                cheat=1
                break
 


print('=====NO=====')
print('BigA')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/ZoomBigA.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('BigA2')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/ZoomBigA2.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('BigA3')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/ZoomBigA3.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('BigA4')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/ZoomBigA4.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('BigA24')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/ZoomBigA24.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('SmallA')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/ZoomSmallA.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('SmallA2')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/ZoomSmallA2.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('SmallA3')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/ZoomSmallA3.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('SmallA4')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/ZoomSmallA4.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('SmallA5')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/ZoomSmallA5.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('SmallA24')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/ZoomSmallA24.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('SmallA25')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/ZoomSmallA25.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('NO1')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO1.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('NO3')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO3.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('NO4')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO4.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('NO6')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO6.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('NO7')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO7.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('NO9')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO9.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('NO10')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO10.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
'''
#당연히안됨
print('NO10_ano')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO10_ano.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
'''
print('NO12')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO12.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
'''
#당연히안됨
print('NO21')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO21.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
'''
print('NO21_ano')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO21_ano.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('NO22')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO22.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('NO23')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO23.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('NO24')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO24.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)





print('=====OK=====')
print('f1')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/Zoomf1.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('f2')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/Zoomf2.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('f3')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/Zoomf3.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('f4')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/Zoomf4.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('f5')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/Zoomf5.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('f24')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/Zoomf24.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('f25')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Long/Zoomf25.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)

print('OK1')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK1.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('OK3')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK3.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('OK4')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK4.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('OK6')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK6.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('OK7')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK7.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('OK9')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK9.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('OK10')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK10.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('OK10_ano')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK10_ano.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('OK12')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK12.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('OK21')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK21.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('OK21_ano')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK21_ano.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('OK22')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK22.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('OK22_ano')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK22_ano.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('OK23')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK23.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)
print('OK24')
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK24.txt', delimiter=',', dtype=np.float32)
detect(predict_fp)

