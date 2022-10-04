import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque

#https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough?hl=ko#%EB%AA%A8%EB%8D%B8_%ED%83%80%EC%9E%85_%EC%84%A0%EC%A0%95


train_dataset_fp = "C:/Users/inoh9/Desktop/ML/Train1_12_23.txt"

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

pyo1 = []
pyo2=[]
pyo3=[]
pyo4=[]
pyo5=[]

for i in range(1, 6):
    for banb in range(1, 50):
        
        if (i==1):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(banb*10, activation=tf.nn.relu, input_shape=(10,)),  # 입력의 형태가 필요합니다.
                tf.keras.layers.Dense(2)
            ])
        if (i==2):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(banb*10, activation=tf.nn.relu, input_shape=(10,)),  # 입력의 형태가 필요합니다.
                tf.keras.layers.Dense(banb*10, activation=tf.nn.relu, input_shape=(10,)),
                tf.keras.layers.Dense(2)
            ])
        if (i==3):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(banb*10, activation=tf.nn.relu, input_shape=(10,)),  # 입력의 형태가 필요합니다.
                tf.keras.layers.Dense(banb*10, activation=tf.nn.relu, input_shape=(10,)),
                tf.keras.layers.Dense(banb*10, activation=tf.nn.relu, input_shape=(10,)),
                tf.keras.layers.Dense(2)
            ])
        if (i==4):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(banb*10, activation=tf.nn.relu, input_shape=(10,)),  # 입력의 형태가 필요합니다.
                tf.keras.layers.Dense(banb*10, activation=tf.nn.relu, input_shape=(10,)),
                tf.keras.layers.Dense(banb*10, activation=tf.nn.relu, input_shape=(10,)),
                tf.keras.layers.Dense(banb*10, activation=tf.nn.relu, input_shape=(10,)),
                tf.keras.layers.Dense(2)
            ])
        if (i==5):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(banb*10, activation=tf.nn.relu, input_shape=(10,)),  # 입력의 형태가 필요합니다.
                tf.keras.layers.Dense(banb*10, activation=tf.nn.relu, input_shape=(10,)),
                tf.keras.layers.Dense(banb*10, activation=tf.nn.relu, input_shape=(10,)),
                tf.keras.layers.Dense(banb*10, activation=tf.nn.relu, input_shape=(10,)),
                tf.keras.layers.Dense(banb*10, activation=tf.nn.relu, input_shape=(10,)),
                tf.keras.layers.Dense(2)
            ])


        predictions = model(features)
        #print(predictions[:5])

        tf.nn.softmax(predictions[:5])


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

        optimizer.apply_gradients(zip(grads, model.trainable_variables))


        train_loss_results = []
        train_accuracy_results = []

        num_epochs = 1

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


            if (i==1):
                pyo1.insert(banb, epoch_accuracy.result())
            if (i==2):
                pyo2.insert(banb, epoch_accuracy.result())
            if (i==3):
                pyo3.insert(banb, epoch_accuracy.result())
            if (i==4):
                pyo4.insert(banb, epoch_accuracy.result())  
            if (i==5):
                pyo5.insert(banb, epoch_accuracy.result()) 

            if epoch % 1 == 0:
                print("에포크 {:03d}: 손실: {:.3f}, 정확도: {:.3%}".format(epoch,
                                                                            epoch_loss_avg.result(),
                                                                            epoch_accuracy.result()))
            
plt.xlabel("Number")
plt.ylabel("Accuracy")
plt.subplot(511)
plt.plot(pyo1)
plt.subplot(512)
plt.plot(pyo2)
plt.subplot(513)
plt.plot(pyo3)
plt.subplot(514)
plt.plot(pyo4)
plt.subplot(515)
plt.plot(pyo5)
plt.show()

'''
fig, axes = plt.subplots(1, sharex=True, figsize=(12, 8))
fig.suptitle('Train')

axes[0].set_ylabel("Accuracy", fontsize=14)
axes[0].set_xlabel("Number", fontsize=14)
axes[0].plot(pyo)

plt.show()
'''


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
