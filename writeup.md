# Getting the data
SHow car and non car

# Feature extraction

I use 3 kinds of features (point to the code for each)

explain how I got to the final parameters of each

show representation of each set of features for the final used configs

# Classifier training

I used SVM...

# Sliding window
I used the simple implementation of the sliding window, slidin the window
across the whole image

I did not use the single HOG extraction because it gave different results
it had less detections

To mitigate the speed issued I used the multiprocessing module to speed up the
processing of the video frames

I started with a window size of
and a scale interval of [x, 3] with 4 scales within that interval


# detections in single images

show examples of detections in single images

show also false positives I guess

# video implementation

## initial strategy
The main idea was to have a thresholded heatmap constructed from several
continguous frames and then use the scipy.ndimage.measurements.label() to Identify
individual blobs in the heatmap. I considered those blobs cars

Show examples of images with all detections, corresponding heatmap and label()

## final strategy
I then moved on to the idea of using...


# discussion

Even with multiprocessing this implementation is not real time. The best I could
get was just under 1 second for the processing of 1 frame. Using a single HOG
per scale would defintely help making it more real-time friendly.

My implementation seems to fail when car is at the very right edge of the image
and on places with shadows, having lots of false positives.

My implementation does not use the fact that the blobs in the heatmap are
persistent objects in the real world. Using this information to somehow model
where the objects would appear in the next frames based on previous frames would
make the implementation more robust. In a real life implementation, we would
probably want to track both the speed of our car and that of the cars surrounding
us to make good predictions.
