import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
import pandas as pd


#https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough?hl=ko#%EB%AA%A8%EB%8D%B8_%ED%83%80%EC%9E%85_%EC%84%A0%EC%A0%95


train_dataset_fp = "C:/Users/inoh9/Desktop/ML/8de2/pyozunha.txt"


column_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', 'abnormal']
'''
column_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28',
'29', '30', '31', '32', '33', '34', '35', '36', 'abnormal']
'''
feature_names = column_names[:-1]
label_name=column_names[-1]

class_names = ['OK', 'Abnormal']

batch_size = 20

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

#print(features)
'''
model = tf.keras.Sequential([
  tf.keras.layers.Dense(20, activation=tf.nn.relu, input_shape=(10,)),  # 입력의 형태가 필요합니다.
  tf.keras.layers.Dense(20, activation=tf.nn.relu),
  tf.keras.layers.Dense(20, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
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
  tf.keras.layers.Dense(50, activation=tf.nn.relu, input_shape=(28,)),  # 입력의 형태가 필요합니다.
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

num_epochs = 51

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

  if epoch % 10 == 0:
    if (epoch != 0):
      print("에포크 {:03d}: 손실: {:.3f}, 정확도: {:.3%}".format(epoch,
                                                                  epoch_loss_avg.result(),
                                                                  epoch_accuracy.result()))


'''
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Train')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()
'''


'''
predict_fp = "C:/Users/inoh9/Desktop/ML/NoLabel.txt"
predict_dataset = tf.data.experimental.make_csv_dataset(
    predict_fp,
    batch_size,
    column_names=column_names[:-1],
    num_epochs=1)
'''

#model.save('C:/Users/inoh9/Desktop/ML/model_pyozunha7')