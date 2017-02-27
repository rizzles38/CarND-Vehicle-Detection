##Writeup Template

---

**Vehicle Detection Project**

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/sliding_window.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in utils.py in the `compute_hog()` function.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an
example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters
(`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random
images from each of the two classes and displayed them to get a feel for what
the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of
`orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled on `pixels_per_cell` of
(8, 8) and `cells_per_block` of (2, 2).  This made the grid a nice power of 2 to
match the image dimensions and seemed to work well for the sizes of cars I was
trying to extract features from.  I settled on `orientations` of 9 which seemed
to give enough discrimination in orientation direction without overfitting to
represent too specific of a shape.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using sklearn.  The data preparation and training functions
are in `utils.py`, named `prep_training_data` and `train_classifier` respectively.

I concatented spatial binning features, color histogram features, and HOG
features all into one large feature vector of 8460 features.  All of these features
were in the YCrCb color space.  After making sure the data was normalized with
`StandardScaler`, I split and shuffled the data into a training set of 80% of
the images and a test set of 20% of the images.

The main training happens in `train.py`, which calls the functions to prep the
data and train the classifier.  Depending on random initialization, the linear
SVM achieves between 99.3% to 99.5% accuracy on the test set.

Finally, `train.py` saves the classifier and `StandardScaler` into a pickle file
so I can just load the classifier and start using it when I run detection.  I only
need to re-run the training step if I want to make a change to how the classifier
is trained.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window generation is done in `utils.py` in the `all_windows` function,
which calls the `sliding_window` function provided from the lectures.  At first
I started with four different scales: 32x32, 64x64, 128x128, and 256x256, but
I ended up dropping the 32x32 size since realistically it was too small to help
much.

I focused the sliding windows at the horizon and below.  The smaller 64x64 windows
stay near the horizon since smaller cars will be further away and closer to the
horizon, whereas the 256x256 windows generally focus all the way to the bottom
of the image.

One optimization I made was to only search in the the lane in front of the car
and to the right.  Since the car is in the far left lane, there's no reason to
search the far left side of the image for detections.

I originally used an overlap of 0.5, but I had one trouble spot where the white
car sat in a particularly unlucky spot straddling a window, so it wasn't in any
window enough to be detected for several seconds.  I fixed this by overlapping
the windows by 0.75 to better increase the chance that a car would be mostly
inside a window for detection.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus
spatially binned YCrCb and histograms of YCrCb in the feature vector, which
provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

