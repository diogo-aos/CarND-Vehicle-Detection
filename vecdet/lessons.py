import time

import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, tuple(size)).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=[64, 64], xy_overlap=[0.5, 0.5],
                    min_overlap=[0.2, 0.5]):
    h, w = img.shape[:2]
    ww, wh = xy_window
    xo, yo = xy_overlap

    x_step = int((1 - xo) * ww)  # step in x axis to achieve desired overlap
    y_step = int((1 - yo) * wh)

    windows = []
    xstart, xstop = x_start_stop
    ystart, ystop = y_start_stop

    # minimum overlap at the edges to add another window
    min_xo = int(min_overlap[0] * ww)
    min_yo = int(min_overlap[1] * wh)

    top = ystart
    while top < ystop - wh:
        left = xstart
        while left < xstop - ww:
            top_left = (left, top)
            bot_right = (left+ww, top+wh)
            windows.append((top_left, bot_right))
            left += x_step
        # check if slack at the right edge > than minimum,add another window
        if xstop - bot_right[0] >= min_xo:
            top_left = (xstop - ww, top_left[1])
            bot_right = (xstop, bot_right[1])
            windows.append((top_left, bot_right))
        top += y_step

    # check if slack at the bottom edge > than minimum,add another row
    top = ystop - wh
    if ystop - (top + wh) >= min_yo:
        left = xstart
        while left < xstop - ww:
            top_left = (left, top)
            bot_right = (left+ww, top+wh)
            windows.append((top_left, bot_right))
            left += x_step
        # check if slack at the right edge > than minimum, add another window
        if xstop - bot_right[0] >= min_xo:
            top_left = (xstop - ww, top_left[1])
            bot_right = (xstop, bot_right[1])
            windows.append((top_left, bot_right))

    return windows


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def convert_color(image, color_space):
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)
    return feature_image

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True,
                        *args, **kwargs):
    # Create a list to append feature vectors to
    features = []
    image = img
    image = img.astype(np.float32)/255
    # apply color conversion if other than 'RGB'
    feature_image = convert_color(image, color_space)

    # spatial_time = time.time()
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=tuple(spatial_size))
        features.append(spatial_features)
    # spatial_time = time.time() - spatial_time
    # hist_time = time.time()
    if hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        features.append(hist_features)
    # hist_time = time.time() - hist_time
    # hog_time = time.time()
    if hog_feat == True:
    # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # hog_time = time.time() - hog_time
    # print('spatial time:', spatial_time)
    # print('hist time:', hist_time)
    # print('hog time:', hog_time)
    # print(hog_time/(hog_time + spatial_time + hist_time))
    features = np.concatenate(features)
    # Return list of feature vectors
    return features


def extract_features_files(imgs_fn, *args, **kwargs):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for fn in imgs_fn:
        file_features = []
        # Read in each one by one
        img = cv2.imread(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_features = extract_features(img, *args, **kwargs)
        yield im_features


def create_heatmap(img, windows, threshold=0):
    heatmap = np.zeros((img.shape[0], img.shape[1]))
    for window in windows:
        heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1
    return apply_threshold(heatmap, threshold)

def apply_threshold(img, threshold):
    im = img.copy()
    im[im <= threshold] = 0
    return im


def draw_labeled_bboxes(img, labels):
    im = img.copy()
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        print('bbox=', bbox)
        # Draw the box on the image
        cv2.rectangle(im, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return im


def get_labeled_bboxes(labels):
    bboxes = []
    # Iterate through all detected boxes
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
    return bboxes

def get_bboxes_from_heatmap(heatmap):
    labels = label(heatmap)
    # im_disp = draw_labeled_bboxes(im_disp, labels)
    return get_labeled_bboxes(labels)
