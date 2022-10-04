from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt


train_dataset_fp = "C:/Users/inoh9/Desktop/ML/LastPredtx7.csv"

column_names = ['Count', 'Change', 'Max_out', 'Max_in', '2sec_out','5sec_out', '10sec_in', 'abnormal']
pre_names = ['Count', 'Change', 'Max_out', 'Max_in', '2sec_out', '5sec_out', '10sec_in']

feature_names = column_names[:-1]
label_name=column_names[-1]

class_names = ['OK', 'Abnormal']

batch_size = 80

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


print('피처:\n', features)
print("라벨\n", labels)
'''
poly=PolynomialFeatures(degree=2)
poly.fit(features)
poly_ftr=poly.transform(features)
'''
poly_ftr=PolynomialFeatures(degree=2).fit_transform(features)
print('변환된 2차 다항식 계수 피처:\n', poly_ftr)
print(poly_ftr.shape)
print(labels.shape)

arl = np.array(labels)
print(arl)
arl = arl.reshape(-1, 1)
print(arl.shape)
poly_FL = np.hstack((poly_ftr, arl))
ab = np.mat(poly_FL)
np.savetxt('pyozunha7.txt', ab, fmt='%d', delimiter=', ')
