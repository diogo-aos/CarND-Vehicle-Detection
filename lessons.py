import matplotlib.image as mpimg
import numpy as np
import cv2
import time
from skimage.feature import hog

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
                    xy_window=[64, 64], xy_overlap=[0.5, 0.5]):
    # If x and/or y start/stop positions not defined, set to image size
    x_start_stop[0] = x_start_stop[0] or 0
    x_start_stop[1] = x_start_stop[1] or img.shape[1]
    y_start_stop[0] = y_start_stop[0] or 0
    y_start_stop[1] = y_start_stop[1] or img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # check area edges
    # if leftover pixels in x axis, add windows touching right edge
    # search_width = x_start_stop[1] - x_start_stop[0]
    # x_leftover_px = search_width % xy_window[0]
    # if x_leftover_px > 0:
    #     y_start = y_start_stop[0]
    #     left = x_start_stop[0] - xy_window[0]
    #     while y_start <= y_start_stop[1] - xy_window[1]:
    #         top_left = (left, y_start)
    #         bot_right = (y_start + xy_window[1], x_start_stop[1])
    #
    #         y_start += ny_pix_per_step
    #
    # # if leftover pixels in y axis, add window touching bottom edge
    # search_height = y_start_stop[1] - y_start_stop[0]
    # y_leftover_px = search_height % xy_window[1]
    # if y_leftover_px > 0:
    #     x_start = x_start_stop[0]
    #     top = y_start_stop[0] - xy_window[1]
    #     while x_start <= x_start_stop[1] - xy_window[0]:
    #         top_left = (x_start, top)
    #         bot_right = (x_start + xy_window[1], y_start_stop[1])
    #
    #         x_start += nx_pix_per_step
    # Return the list of windows
    return window_list

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
                        spatial_feat=True, hist_feat=True, hog_feat=True):
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
