import pickle
import json
import sys
import fnmatch
import os
import time
import multiprocessing as mp
from collections import OrderedDict
from random import shuffle

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

if '-o' in sys.argv:
    idx = sys.argv.index('-o')
    output_dir = sys.argv[idx + 1]
else:
    output_dir = ''


cars_dataset_paths = train_config['car_dataset']
not_cars_dataset_paths = train_config['noncar_dataset']

# cars_root_fn = '/home/chiro/workspace/self_driving_car/CarND-Vehicle-Detection/dataset/vehicles/'
# not_cars_root_fn = '/home/chiro/workspace/self_driving_car/CarND-Vehicle-Detection/dataset/non-vehicles/'

cars = []
for cars_root_fn in cars_dataset_paths:
    for root, dirnames, filenames in os.walk(cars_root_fn):
        for filename in fnmatch.filter(filenames, '*.png'):
            cars.append(os.path.join(root, filename))

not_cars = []
for not_cars_root_fn in not_cars_dataset_paths:
    for root, dirnames, filenames in os.walk(not_cars_root_fn):
        for filename in fnmatch.filter(filenames, '*.png'):
            not_cars.append(os.path.join(root, filename))

print('cars length:', len(cars))
print('not cars length:', len(not_cars))

shuffle(cars)
shuffle(not_cars)

# cars = cars[:500]
# not_cars = not_cars[:500]

# cars = cars[::2]

print('training with {} car samples and {} not car sampels'.format(len(cars), len(not_cars)))

all_fn = cars.copy()
all_fn.extend(not_cars)

import tempfile
# 4 processes, one for half of each set
def extract(lst, q):
    fn = tempfile.mktemp()
    feature_gen = extract_features_files(lst, **train_config)
    features = list(feature_gen)
    with open(fn, 'wb') as f:
        pickle.dump(features, f)
    q.put(fn)
    print('done; saved data in ', fn)

def extract_progbar(lst, q):
    fn = tempfile.mktemp()
    feature_gen = extract_features_files(lst, **train_config)
    features = []
    progress = tqdm.tqdm(total=len(lst))
    for feature in feature_gen:
        features.append(feature)
        progress.update()
    with open(fn, 'wb') as f:
        pickle.dump(features, f)
    q.put(fn)
    # print('done; saved data in ', fn)

feature_ext_time = time.time()

n_procs = 4
steps = list(np.linspace(0, len(all_fn), n_procs+1, dtype=np.int))
queues = [mp.Queue() for x in range(n_procs)]
procs = []
for i, q in enumerate(queues):
    func = extract
    if i == 0:
        func = extract_progbar
    p = mp.Process(target=func, args=(all_fn[steps[i]:steps[i+1]], q))
    p.start()
    procs.append(p)

temp_files = [q.get(block=True) for q in queues]
# print(temp_files)

feature_ext_time = round(time.time() - feature_ext_time, 2)
print('took {} seconds to extract all features'.format(feature_ext_time))

all_features = []
for fn in temp_files:
    with open(fn, 'rb') as f:
        all_features.extend(pickle.load(f))


# Create an array stack of feature vectors
X = np.vstack(tuple(all_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(cars)), np.zeros(len(not_cars))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Feature vector length:', len(X_train[0]))
print('training classifier...')
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

classifier_fn = '{}_classifier.p'.format(train_name)
metadata_fn = '{}_metadata.json'.format(train_name)
classifier_config_fn = '{}_cv_config.json'.format(train_name)
classifier_run_config_fn = '{}_run_config.json'.format(train_name)

classifier_fn = os.path.join(output_dir, classifier_fn)
metadata_fn = os.path.join(output_dir, metadata_fn)
classifier_config_fn = os.path.join(output_dir, classifier_config_fn)
classifier_run_config_fn = os.path.join(output_dir, classifier_run_config_fn)

with open(classifier_fn, 'wb') as f:
    pickle.dump({'classifier': svc, 'scaler': X_scaler}, f)

with open(classifier_config_fn, 'w') as f:
    json.dump(train_config, f, indent=2)

with open('run_config_default.json', 'r') as f:
    run_config_default = json.load(f, object_pairs_hook=OrderedDict)

run_config_default['scaler_classifier_path'] = classifier_fn
run_config_default['train_config_path'] = classifier_config_fn

with open(classifier_run_config_fn, 'w') as f:
    json.dump(run_config_default, f, indent=2)

metadata = OrderedDict({
    'n_cars': len(cars),
    'n_not_cars': len(not_cars),
    'n_features': len(X_train[0]),
    'feature_extraction_time': feature_ext_time,
    'train_size': len(X_train),
    'test_size': len(X_test),
    'accuracy': accuracy,
    'train_time': train_time,
    'predict_10_time': predict_10_time,
    'comment': '',
})

with open(metadata_fn, 'w') as f:
    json.dump(metadata, f, indent=2)

print('saved classifier and scaler in {}'.format(classifier_fn))
print('saved classifier metadata in {}'.format(metadata_fn))
print('saved classifier config in {}'.format(classifier_config_fn))
print('saved classifier default run config in {}'.format(classifier_run_config_fn))
