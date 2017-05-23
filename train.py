from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from lessons import extract_features_files

import numpy as np

import pickle
import json

import sys
import fnmatch
import os

import time
import tqdm

if '--train-config' in sys.argv:
    idx = sys.argv.index('--train-config')
    train_config_fn = sys.argv[idx + 1]
    with open(train_config_fn, 'r') as f:
        train_config = json.load(f)
else:
    print('missing --train-config')
    sys.exit(0)


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

print('cars length:', len(cars))
print('not cars length:', len(not_cars))

# cars = cars[:500]
# not_cars = not_cars[:500]

car_features_gen = extract_features_files(cars, **train_config)
not_car_features_gen = extract_features_files(not_cars, **train_config)

car_features = []
not_car_features = []

print('extracting cars...')
for feature in tqdm.tqdm(car_features_gen, total=len(cars)):
    car_features.append(feature)

print('extracting not cars...')
for feature in tqdm.tqdm(not_car_features_gen, total=len(not_cars)):
    not_car_features.append(feature)

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
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


classifier_fn = 'svc_classifier.p'
with open(classifier_fn, 'wb') as f:
    json.dump(f, {'classifier': svc, 'scaler': X_scaler})

print('saved classifier and scaler in {}'.format(classifier_fn))
