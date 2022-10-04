from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import tensorflow as tf

# 다항식으로 변환한 단항식 생성, [[0,1],[2,3]]의 2X2 행렬 생성
#X = np.arange(4).reshape(2,2)
#print('일차 단항식 계수 feature:\n',X )

X = [2, 3, 5, 7, 11, 13, 17]
detect_data = tf.convert_to_tensor([X])
poly_ftr=PolynomialFeatures(degree=2).fit_transform(detect_data)

# degree = 2 인 2차 다항식으로 변환하기 위해 PolynomialFeatures를 이용하여 변환
print('변환된 2차 다항식 계수 feature:\n', poly_ftr)