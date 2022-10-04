# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# Function to create model, required for KerasClassifier
'''
def create_model():
	# create model
    
	model = Sequential()
	model.add(Dense(32, input_dim=10, activation='relu'))
	model.add(Dense(32, activation='relu'))

	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    

	return model
'''
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(50, activation='relu', input_shape=(28,)),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam',
                    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("featu.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:28]
Y = dataset[:,28]
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#epochs = [1, 2, 3, 4, 5]
epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))