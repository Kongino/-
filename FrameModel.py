import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt

#https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough?hl=ko#%EB%AA%A8%EB%8D%B8_%ED%83%80%EC%9E%85_%EC%84%A0%EC%A0%95


train_dataset_fp = "C:/Users/inoh9/Desktop/ML/Train8_2.csv"

#column_names = ['eye', 'nose', 'ear', 'abnormal']

column_names = ['NoseX', 'NoseY', 'ReyeX', 'ReyeY', 'LeyeX', 'LeyeY', 'RearX', 'RearY', 'LearX', 'LearY', 'abnormal']
pre_names = ['NoseX', 'NoseY', 'ReyeX', 'ReyeY', 'LeyeX', 'LeyeY', 'RearX', 'RearY', 'LearX', 'LearY']

feature_names = column_names[:-1]
label_name=column_names[-1]

class_names = ['OK', 'Abnormal']

batch_size = 50

train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1
)

features, labels = next(iter(train_dataset))


def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))
#print(features[:5])
'''
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(10,)),  # 입력의 형태가 필요합니다.
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(2)
])
'''
'''
model = tf.keras.Sequential([
  tf.keras.layers.Dense(300, activation=tf.nn.relu, input_shape=(10,)),  # 입력의 형태가 필요합니다.
  tf.keras.layers.Dense(300, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2)
])
'''

model = tf.keras.Sequential([
  tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(10,)),  # 입력의 형태가 필요합니다.
  tf.keras.layers.Dense(32, activation=tf.nn.relu),
  tf.keras.layers.Dense(2)
])

predictions = model(features)
#print(predictions[:5])

tf.nn.softmax(predictions[:5])

#print("예측 : {}".format(tf.argmax(predictions, axis=1)))
#print("레이블 : {}".format(labels))

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y):
  y_ = model(x)

  return loss_object(y_true=y, y_pred=y_)


l = loss(model, features, labels)
#print("손실 테스트: {}".format(l))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_value, grads = grad(model, features, labels)
'''
print("단계: {}, 초기 손실: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))
'''
optimizer.apply_gradients(zip(grads, model.trainable_variables))
'''
print("단계: {},      손실: {}".format(optimizer.iterations.numpy(),
                                          loss(model, features, labels).numpy()))
'''

## 노트: 이 셀을 다시 실행하면 동일한 모델의 변수가 사용됩니다.

# 도식화를 위해 결과를 저장합니다.
train_loss_results = []
train_accuracy_results = []

num_epochs = 6

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # 훈련 루프 - 32개의 배치를 사용합니다.
  for x, y in train_dataset:
    # 모델을 최적화합니다.
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # 진행 상황을 추적합니다.
    epoch_loss_avg(loss_value)  # 현재 배치 손실을 추가합니다.
    # 예측된 레이블과 실제 레이블 비교합니다.
    epoch_accuracy(y, model(x))

  # epoch 종료
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 1 == 0:
    print("에포크 {:03d}: 손실: {:.3f}, 정확도: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

'''
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Train')

axes[0].set_ylabel("loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("accuracy", fontsize=14)
axes[1].set_xlabel("epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()
'''

test_fp = "C:/Users/inoh9/Desktop/ML/Test8_2.csv"

test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
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


'''
predict_fp = "C:/Users/inoh9/Desktop/ML/NoLabel.txt"
predict_dataset = tf.data.experimental.make_csv_dataset(
    predict_fp,
    batch_size,
    column_names=column_names[:-1],
    num_epochs=1)
'''
#괜찮은거 하나 저장함.
model.save('C:/Users/inoh9/Desktop/ML/model_testreinfor')

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

  for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    if (class_idx == 1):
      formax_in=0
      count=count+1
      if (befo == 0):
        chcount=chcount+1
      else :
        formax_out=formax_out+1
        for2sec=for2sec+1
      befo=1
    else :
      befo = 0
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
  if (no == 0) :
    print("PredOK{:<6} - Count {:<6} Change Count {:<6} max_out {:<6} max_in {:<6} 2sec_out {:<6}".format(n, count, chcount, max_out, max_in, more_2sec))
  else :
    print("PredNO{:<6} - Count {:<6} Change Count {:<6} max_out {:<6} max_in {:<6} 2sec_out {:<6}".format(n, count, chcount, max_out, max_in, more_2sec))


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

#여기부터는 의미없음
'''
predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK1.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredOK1 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK3.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredOK3 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK4.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredOK4 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK6.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredOK6 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK7.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredOK7 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK10.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredOK10 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK12.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredOK12 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK21.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredOK21 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK21_ano.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredOK21_ano - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK10_ano.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredOK10_ano - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK22.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredOK22 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredOK23.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredOK23 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO1.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredNO1 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO3.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredNO3 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO4.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredNO4 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO6.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredNO6 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO7.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredNO7 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO10.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredNO10 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO12.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredNO12 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO21.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredNO21 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO10_ano.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredNO10_ano - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO22.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredNO22 - Count {}, Change Count {}".format(count, chcount))

predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/ZoomPredNO23.txt', delimiter=',', dtype=np.float32)
#predict_dataset = tf.convert_to_tensor(predict_fp)


predictions = model(predict_fp)

count=0
befo=0
chcount=0

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  if (class_idx == 1):
    count=count+1
    if (befo == 0):
      chcount=chcount+1
    befo=1
  else :
    befo = 0

print("PredNO23 - Count {}, Change Count {}".format(count, chcount))
'''


'''

predict_dataset = tf.convert_to_tensor([
    [22, 15, 17],
    [88, 56, 67],
    [48, 31, 15]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("샘플 {} 예측: {} ({:4.1f}%)".format(i, name, 100*p))

'''

