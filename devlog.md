# current state
2017-05-30 08:37

I have a pipeline working OK. The "normal" pipeline (extracting HOG for each
window) has slightly different results than the "hog-once" pipeline - I don't
know why yet.

There are 2 configuration files: train_config.json configures the training
of the classifier and the CV processing; run_config.json configures parameters
for the pipeline, e.g. window size, window overlap, area to search, etc.

I'm currently training with all the data that is available. The classes are
balanced. I've read about **hard-negative mining** (putting false positives in
the negative training set), but have not tried it yet. I also wonder what would
be acceptable to use for these and not be considered cheating. I also read
[here](http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/) that the #negative samples should be >> #positive samples.


# creating class imbalance
2017-05-30 21:22

I reduced the car dataset to 25%. The classifier accuracy reduced ~2%. The
results on the test images seem encouraging. There are more false positives but
even more true positives, which allow for a better thresholding. There is only
one image where I can't get a good heatmap with a threshold of 3. I think doing
the heatmap over contiguous frames will help.

# checkpoint 1
2017-05-31 08:52

Still using the same class imbalance, I allowed windows from 64x64 to 3 times
that size. Using a threshold of 3, I got good heatmaps on all test images. The
training and run files for these checkpoint are saved in /trainings/checkpoint1.

# problems with video frames processing
2017-05-31 21:24

The process going through the video frames of the project video is getting
killed midway and I don't know why. I need to see if it's consuming too much
memory or something like that.

I should also parallelize the pipeline (one scale per process). I also want to
reduce the spatial features to 16x16 and see the effect.


# too many false positives
2017-06-10 17:15

I think using only 25% of the positive dataset is a bit too little. I'll train
a new classifier with 50%.

Isabella also suggested processing the dataset by hand to remove contiguous
frames from the dataset.

# where to improve?
2017-06-11 21:24

I trained the new classifier. The performance increased a little, but I still
get a lot of false positives. I'm thinking about trying 2 things:

1. increase minimum window size
2. train new classifier without the data that has "redundant" frames

# tracking and removing false positives
2017-06-12 07:33

Tracking: when a detection would exceed a second threshold or if it is persistent
for enough frames, it could be considered a car and we could then try to model
its movement from frame to frame. I thought about template matching, but this
would be no good when the car goes under shadows.

Removing false positives: I need a more robust way, a second layer perhaps, of
removing false positives. They seem to be more jumpy from frame to frame, so
maybe that's something I could use. Maybe something like mainting the heatmap
buffer but now base is off a second heatmap that tracks heat from frame to
frame.
