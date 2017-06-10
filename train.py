from random import shuffle
import pickle
import json
import sys
import fnmatch
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import tqdm

from lessons import extract_features_files
from utils import *

train_config = get_train_config_from_cli(sys.argv)

if '--train-name' in sys.argv:
    idx = sys.argv.index('--train-name')
    train_name = sys.argv[idx + 1]
else:
    train_name = 'classifier'


cars_root_fn = '/home/chiro/workspace/self_driving_car/CarND-Vehicle-Detection/dataset/vehicles/'
not_cars_root_fn = '/home/chiro/workspace/self_driving_car/CarND-Vehicle-Detection/dataset/non-vehicles/'


cars = []
for root, dirnames, filenames in os.walk(cars_root_fn):
    for filename in fnmatch.filter(filenames, '*.png'):
        cars.append(os.path.join(root, filename))

not_cars = []
for root, dirnames, filenames in os.walk(not_cars_root_fn):
    for filename in fnmatch.filter(filenames, '*.png'):
        not_cars.append(os.path.join(root, filename))


# cars = cars[:500]
# not_cars = not_cars[:500]

shuffle(cars)
cars = cars[::2]

print('cars length:', len(cars))
print('not cars length:', len(not_cars))

car_features_gen = extract_features_files(cars, **train_config)
not_car_features_gen = extract_features_files(not_cars, **train_config)

car_features = []
not_car_features = []

feature_ext_time = time.time()

progress = tqdm.tqdm(total=len(cars) + len(not_cars))
print('extracting cars...')
for feature in car_features_gen:
    car_features.append(feature)
    progress.update()

print('extracting not cars...')
for feature in not_car_features_gen:
    not_car_features.append(feature)
    progress.update()

feature_ext_time = round(time.time() - feature_ext_time, 2)

# Create an array stack of feature vectors
X = np.vstack((car_features, not_car_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
train_time=time.time()
svc.fit(X_train, y_train)
train_time = round(time.time() - train_time, 2)
print(train_time, 'Seconds to train SVC...')
# Check the score of the SVC
accuracy = round(svc.score(X_test, y_test), 4)
print('Test Accuracy of SVC = ', accuracy)
# Check the prediction time for a single sample
predict_10_time = time.time()
n_predict = 10
predict_10_time = round(time.time() - predict_10_time, 2)

print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
print(round(predict_10_time, 5), 'Seconds to predict',
      n_predict,'labels with SVC')


classifier_fn = '{}_classifier.p'.format(train_name)
metadata_fn = '{}_metadata.json'.format(train_name)
classifier_config_fn = '{}_cv_config.json'.format(train_name)

with open(classifier_fn, 'wb') as f:
    pickle.dump({'classifier': svc, 'scaler': X_scaler}, f)
metadata = {
    'accuracy': accuracy,
    'train_size': len(X_train),
    'test_size': len(X_test),
    'n_features': len(X_train[0]),
    'feature_extraction_time': feature_ext_time,
    'train_time': train_time,
    'predict_10_time': predict_10_time
}

with open(metadata_fn, 'w') as f:
    json.dump(metadata, f, indent=2)

with open(classifier_config_fn, 'w') as f:
    json.dump(train_config, f, indent=2)

print('saved classifier and scaler in {}'.format(classifier_fn))
print('saved classifier metadata in {}'.format(metadata_fn))
print('saved classifier config in {}'.format(classifier_config_fn))
