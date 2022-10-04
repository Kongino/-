import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt

Pyo_fp = "C:/Users/inoh9/Desktop/ML/Forgraph2.txt"

#column_names = ['eye', 'nose', 'ear', 'abnormal']

column_names = ['Count', 'Change', 'Abnormal']

feature_names = column_names[:-1]
label_name=column_names[-1]

class_names = ['OK', 'Abnormal']

batch_size = 80

train_dataset = tf.data.experimental.make_csv_dataset(
    Pyo_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1
)

features, labels = next(iter(train_dataset))

plt.scatter(features['Count'],
            features['Change'],
            c=labels,
            cmap='viridis')

plt.xlabel("Count")
plt.ylabel("Change")
plt.show()
