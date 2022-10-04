import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt

#https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ko


#model = tf.keras.models.load_model('C:/Users/inoh9/Desktop/ML/model_1_21_23_test96')
model = tf.keras.models.load_model('C:/Users/inoh9/Desktop/ML/model_8de2_93')
model.summary()



def printmodel(predic, n, no):
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
  more_10secin=0
  for10secin=0


  for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    if (class_idx == 1):
      for10secin=0
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
      for10secin=for10secin+1
      befo = 0
      formax_out=0
      for2sec=0
      for5sec=0
      formax_in=formax_in+1
    if (formax_out > max_out):
      max_out=formax_out
    if (formax_in > max_in):
      max_in=formax_in
    if (for2sec > 60):
      more_2sec=more_2sec+1
      for2sec=0
    if (for5sec>150):
      more_5sec=more_5sec+1
      for5sec=0
    if (for10secin>300):
      more_10secin=more_10secin+1
      for10secin=0
  if (no == 0) :
    f=open("LastPredtx7.txt", 'a')
    print("PredOK{:<6} - Count {:<6} Change Count {:<6} max_out {:<6} max_in {:<6} 2sec_out {:<6} 5sec_out {:<6} 10sec_in {:<6}".format(n, count, chcount, max_out, max_in, more_2sec, more_5sec, more_10secin))
    if (max_in > 3600):
      max_in=3600
    if (max_out>3600):
      max_out=3600
    data="%d, %d, %d, %d, %d, %d, %d, 0 \n"%(count, chcount*2, max_out, max_in, more_2sec*60, more_5sec*150, more_10secin*300)
    f.write(data)
    f.close()
  else :
    f=open("LastPredtx7.txt", 'a')
    print("PredNO{:<6} - Count {:<6} Change Count {:<6} max_out {:<6} max_in {:<6} 2sec_out {:<6} 5sec_out {:<6} 10sec_in {:<6}".format(n, count, chcount, max_out, max_in, more_2sec, more_5sec, more_10secin))
    if (max_in > 3600):
      max_in=3600
    if (max_out>3600):
      max_out=3600
    data="%d, %d, %d, %d, %d, %d, %d, 1 \n"%(count, chcount*2, max_out, max_in, more_2sec*60, more_5sec*150, more_10secin*300)
    f.write(data)
    f.close()



def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

column_names = ['NoseX', 'NoseY', 'ReyeX', 'ReyeY', 'LeyeX', 'LeyeY', 'RearX', 'RearY', 'LearX', 'LearY', 'abnormal']

test_fp = "C:/Users/inoh9/Desktop/ML/Test1_24_25OK.csv"

test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    50,
    column_names=column_names,
    label_name='abnormal',
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)

test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  logits = model(x)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("테스트 세트 정확도: {:.3%}".format(test_accuracy.result()))


predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK1.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 1, 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK3.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 3, 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK4.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 4, 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK6.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 6, 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK7.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 7, 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK9.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 9, 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK10.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 10, 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK10_ano.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, '10_ano', 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK12.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 12, 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK21.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 21, 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK21_ano.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, '21_ano', 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK22.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 22, 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK23.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 23, 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK22_ano.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, '22_ano', 0)

#이상함
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK23_ano.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, '23_ano', 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK24.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 24, 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK24_ano.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, '24_ano', 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK25.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 25, 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomOK25_ano.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, '25_ano', 0)


#now 'NO'
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO1.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 1, 1)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO3.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 3, 1)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO4.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 4, 1)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO6.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 6, 1)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO7.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 7, 1)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO9.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 9, 1)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO10.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 10, 1)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO10_ano.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, '10_ano', 1)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO12.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 12, 1)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO21.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 21, 1)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO22.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 22, 1)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO23.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 23, 1)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO24.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 24, 1)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO24_ano.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, '24_ano', 1)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO25.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 25, 1)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomNO25_ano.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, '25_ano', 1)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Special/SpecialPredOK2.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 'SpecialOK2', 0)

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/Special/SpecialPredNO2.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 'SpecialNO2', 1)
