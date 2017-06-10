- train_mp.py
  - loads dataset
  - extracts features of all images with multiple processes
  - trains classifier
  - saves with a training name prefix for each
    - used computer vision configuration
    - pre-filled run configuration for the pipeline
    - metadata about the training process, such as run times, dataset size, etc.
    - classifier and scaler
  - `python -W ignore train_mp.py --train-config train_config.json --train-name "class_inbalance" -o trainings`
    - `--train-config` is the computer vision config path
    - `--train-name` is the prefix to the output files
    - `-o` is the output dir

- pipeline_mp.py
  - several functions to process an image
  - processes extracted windows from an image with multiple processes

- video_frames.py
  - takes in images and processes them as if it were the real video
  - saves each processed frame
  - saves all frames detections in a pickled file
  - `python -W ignore video_frames.py '../CarND-Advanced-Lane-Lines/video_imgs/project_video/*.jpg' --run-config trainings/class_inbalance_run_config.json -o output/`
    - `--run-config` is the run configuration path
    - `-o` is the output dir, real output is saved in a crated dir `processed`

- tweak.py
  - reads in the pickled detections
  - visualization of the initial images
  - visualization of the heatmaps
  - main purpose to configure final detection parameters
    - heatmap threshold
    - heatmap buffer size

- utils.py
  - getting computer vision and run configutions from CLI arguments
  - reading an image with OpenCV and converting it to RGB
  - draw text on an image

- lessons.py
  - a bunch of functions from the course lessons, with some minor changes
  - HOG transform
  - color histogram
  - spatial features
  - extract features

- x_cv_config.json
```json
{
  "spatial_feat": true,
  "hist_feat": true,
  "hog_feat": true,
  "color_space": "YCrCb",
  "spatial_size": [
    32,
    32
  ],
  "hist_bins": 32,
  "orient": 9,
  "pix_per_cell": 8,
  "cell_per_block": 2,
  "hog_channel": "ALL"
}
```
  - `{spatial, hist, hog}_feat` specifies whether these features or not
  - `color_space` can be RGB, HLS, HSV, YCrCb
  - `spatial_size` is the resize size to be used in the features
  - `hist_bins` is how many bins to use for the color histograms
  - `orient` is how many bins to use for the HOG
  - `pix_per_cell` and `cell_per_block` are parameters for the HOG
  - `hog_channel` specifies which channel or "ALL" to apply the HOG to

  - x_run_config.json
  ```json
  {
    "buffer_max_len": 5,
    "y_start_end": [
      400,
      656
    ],
    "x_start_end": [
      0,
      1280
    ],
    "window_size": [
      64,
      64
    ],
    "window_scale_min_max": [
      1.0,
      3.0
    ],
    "window_n_scales": 4,
    "window_overlap": [
      0.75,
      0.75
    ],
    "heatmap_min_threshold": 1,
    "scaler_classifier_path": "trainings/class_imbalance_50percent_classifier.p",
    "train_config_path": "trainings/class_imbalance_50percent_cv_config.json"
  }
  ```
    - `buffer_max_len` specifies how many frames to use for the heatmap
    - `{x,y}_start_end` specify a window of the image to use
    - `window_size` specifies the sliding window size
    - `window_scale_min_max` specifies a range of scales of the `window_size`
    to be used
    - `window_n_scales` specifies how many different scales to use in the
    `window_scale_min_max` interval
    - `heatmap_min_threshold` is the minimum acceptable value in the heatmap
    - `scaler_classifier_path` is the path to the classifier
    - `train_config_path` is the path to the computer vision configuration
