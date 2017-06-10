import json
from collections import OrderedDict
import pickle

import cv2

def get_train_config_from_cli(argv):
    if '--train-config' not in argv:
        raise ValueError('train config not in arguments')
    idx = argv.index('--train-config')
    train_config_fn = argv[idx + 1]
    with open(train_config_fn, 'r') as f:
        train_config = json.load(f, object_pairs_hook=OrderedDict)
    return train_config

def get_run_config_from_cli(argv):
    if '--run-config' not in argv:
        raise ValueError('run config not in arguments')
    idx = argv.index('--run-config')
    run_config_fn = argv[idx + 1]
    with open(run_config_fn, 'r') as f:
        run_config = json.load(f, object_pairs_hook=OrderedDict)
    return run_config

def get_stuff_from_run_config(conf):
    with open(conf['scaler_classifier_path'], 'rb') as f:
        classifier_scaler = pickle.load(f)
    with open(conf['train_config_path'], 'r') as f:
        train_config = json.load(f, object_pairs_hook=OrderedDict)
    return classifier_scaler, train_config

def read_image_rgb(fn):
    img = cv2.imread(fn)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def write_on_image(img, txt=[]):
    'write strings in txt in corresponding lines in lines'
    img = img.copy()
    x, y = 0, 35
    for l, t in enumerate(txt):
        cv2.putText(img, t, (0, y * (l + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    return img
