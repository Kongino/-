import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt

#https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough?hl=ko#%EB%AA%A8%EB%8D%B8_%ED%83%80%EC%9E%85_%EC%84%A0%EC%A0%95


train_dataset_fp = "C:/Users/inoh9/Desktop/ML/handplus/handplus_Train.txt"

#column_names = ['eye', 'nose', 'ear', 'abnormal']

column_names = ['NoseX', 'NoseY','NeckX', 'NeckY', 'RShoulderX', 'RShoulderY', 'RElbowX', 'RElbowY',
                    'RWristX', 'RWristY', 'LShoulderX', 'LShoulderY', 'LElbowX', 'LElbowY',
                    'LWristX', 'LWristY', 'ReyeX', 'ReyeY', 'LeyeX', 'LeyeY', 'RearX', 'RearY', 'LearX', 'LearY', 'abnormal']
pre_names = ['NoseX', 'NoseY','NeckX', 'NeckY', 'RShoulderX', 'RShoulderY', 'RElbowX', 'RElbowY',
                    'RWristX', 'RWristY', 'LShoulderX', 'LShoulderY', 'LElbowX', 'LElbowY',
                    'LWristX', 'LWristY', 'ReyeX', 'ReyeY', 'LeyeX', 'LeyeY', 'RearX', 'RearY', 'LearX', 'LearY']

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
  tf.keras.layers.Dense(50, activation=tf.nn.relu, input_shape=(24,)),  # 입력의 형태가 필요합니다.
  tf.keras.layers.Dense(50, activation=tf.nn.relu),
  tf.keras.layers.Dense(50, activation=tf.nn.relu),
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

num_epochs = 4

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

test_fp = "C:/Users/inoh9/Desktop/ML/handplus/handplus_Test.txt"

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
#model.save('C:/Users/inoh9/Desktop/ML/my_model_next')

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


predict_fp = np.loadtxt('C:/Users/inoh9/Desktop/ML/handplus/handplus_PredNO1.txt', delimiter=',', dtype=np.float32)
printmodel(predict_fp, 1, 1)

#now 'NO'
