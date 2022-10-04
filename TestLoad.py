import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt

#https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ko


model = tf.keras.models.load_model('C:/Users/inoh9/Desktop/ML/my_model_test96')

model.summary()

predd = tf.convert_to_tensor([[1133.615479, 409.636627, 1121.984619, 368.368896, 1171.954102, 344.955231, 0.000000, 0.000000, 1310.240356, 365.496429]])
pre = model(predd)
for i, logits in enumerate(pre):
    print(tf.argmax(logits).numpy())

