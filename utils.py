import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

def load_training():
    cars = glob.glob("training/vehicles/**/*.png")
    not_cars = glob.glob("training/non-vehicles/**/*.png")
    return cars, not_cars

def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel().astype(np.float64)
    return features

def color_hist(img, nbins=32):
    features = np.histogram(img, bins=nbins)[0].astype(np.float64)
    return features

def compute_hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), vis=False, transform_sqrt=False):
    if vis:
        features, hog_img = hog(img,
                                orientations=orientations,
                                pixels_per_cell=pixels_per_cell,
                                cells_per_block=(cells_per_block),
                                transform_sqrt=transform_sqrt,
                                visualise=True, feature_vector=True)
        return features, hog_img
    else:
        features = hog(img,
                       orientations=orientations,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=(cells_per_block),
                       transform_sqrt=transform_sqrt,
                       visualise=False, feature_vector=True)
        return features

def extract_features(img):
    feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    # Spatial binning on each channel
    spatials = []
    for channel in range(feature_img.shape[2]):
        spatials.append(bin_spatial(feature_img[:,:,channel]))

    # Color histogram on each channel
    histograms = []
    for channel in range(feature_img.shape[2]):
        histograms.append(color_hist(feature_img[:,:,channel]))

    # HOG features on each channel
    hogs = []
    for channel in range(feature_img.shape[2]):
        hogs.append(compute_hog(feature_img[:,:,channel]))

    # Concatenate the feature vectors
    return np.concatenate((spatials[0], spatials[1], spatials[2],
                           histograms[0], histograms[1], histograms[2],
                           hogs[0], hogs[1], hogs[2]))

# Takes a list of image filenames and returns a list of feature vectors from all
# the images
def feature_vectors(filenames):
    features = []
    for f in filenames:
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        extracted_features = extract_features(img)
        features.append(extracted_features)
    return features

# Takes a big list of feature vectors and computes a sensible normalizer
def get_feature_normalizer(feature_vectors):
    return StandardScaler().fit(feature_vectors)

# Given a feature normalizer and a list of feature vectors you want normalized,
# normalizes the feature vectors using the normalizer and returns them
def normalize_feature_vectors(feature_vectors, normalizer):
    return normalizer.transform(feature_vectors)

# Given a list of car files and not car files, preps the feature data for
# training a support vector machine
def prep_training_data(car_files, not_car_files, test_size=0.2):
    # Get the feature vectors for cars and not cars
    car_features = feature_vectors(car_files)
    not_car_features = feature_vectors(not_car_files)

    # Create label vectors to match the feature vectors
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    # Create an input vector to match by combining all the feature vectors and
    # normalizing them
    all_features = np.vstack([car_features, not_car_features])
    normalizer = get_feature_normalizer(all_features)
    X = normalize_feature_vectors(all_features, normalizer)

    # Shuffle and split the labeled data into training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

    return X_train, X_test, y_train, y_test, normalizer

# Given training and test sets, trains a linear SVM, returns it, and reports the
# accuracy on the test set
def train_classifer(X_train, y_train, X_test, y_test):
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    accuracy = svc.score(X_test, y_test)
    return svc, accuracy

def save_classifier(classifier, normalizer):
    data = {
        "classifier": classifier,
        "normalizer": normalizer,
    }
    pickle.dump(data, open("classifier.pickle", "wb"))

def load_classifier():
    data = pickle.load(open("classifier.pickle", "rb"))
    return data["classifier"], data["normalizer"]

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64,  64), xy_overlap=(0.5, 0.5)):
    # Initialize start/stop positions if not given
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

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
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))

    # Return the list of windows
    return window_list

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def all_windows(img):
    # 64x64 window
    windows64 = slide_window(img,
                             x_start_stop=[500, 1280],
                             y_start_stop=[400, 496],
                             xy_window=(64, 64),
                             xy_overlap=(0.75, 0.75))

    # 128x128 window
    windows128 = slide_window(img,
                              x_start_stop=[500, 1280],
                              y_start_stop=[400, 592],
                              xy_window=(128, 128),
                              xy_overlap=(0.75, 0.75))

    # 256x256 window
    windows256 = slide_window(img,
                              x_start_stop=[500, 1280],
                              y_start_stop=[400, 720],
                              xy_window=(256, 256),
                              xy_overlap=(0.75, 0.75))

    # Uncomment to draw all sliding windows
    #img = draw_boxes(img, windows64, color=(0, 255, 0), thick=2)
    #img = draw_boxes(img, windows128, color=(0, 0, 255), thick=2)
    #img = draw_boxes(img, windows256, color=(255, 255, 0), thick=2)
    #plt.imshow(img)
    #plt.show()
    #exit()

    return windows64 + windows128 + windows256

def search_windows(img, windows, classifier, normalizer):
    on_windows = []
    for window in windows:
        # Extract test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        # Extract features from the test window and normalize
        features = extract_features(test_img)
        normalized = normalize_feature_vectors([features], normalizer)[0]

        # Classify the window
        prediction = classifier.predict(normalized)
        if prediction == 1:
            on_windows.append(window)

    # Return positive detection windows
    return on_windows

def heatmap_image(img, boxes):
    heatmap = np.zeros_like(img[:,:,0])
    for box in boxes:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def filter_heatmaps(heatmaps, threshold=1):
    combined = np.zeros_like(heatmaps[0])
    for heatmap in heatmaps:
        combined = np.add(combined, heatmap)
    combined[combined <= threshold] = 0
    labels = label(combined)
    return labels

def draw_labels(img, labels, color=(0, 0, 255), thick=6):
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)

    return img

def output_plots():
    cars, not_cars = load_training()
    print("Found {} cars.".format(len(cars)))
    print("Found {} not cars.".format(len(not_cars)))

    car = cv2.imread(cars[5000])
    car = cv2.cvtColor(car, cv2.COLOR_BGR2RGB)
    not_car = cv2.imread(not_cars[5000])
    not_car = cv2.cvtColor(not_car, cv2.COLOR_BGR2RGB)

    fig = plt.figure()

    plt.subplot(121)
    plt.imshow(car)
    plt.title("Example Car")

    plt.subplot(122)
    plt.imshow(not_car)
    plt.title("Example Not Car")

    plt.show()

    fig, ax = plt.subplots(3, 4, sharex="col", sharey="row")

    car_ycrcb = cv2.cvtColor(car, cv2.COLOR_RGB2YCrCb)

    ax[0, 0].imshow(car_ycrcb[:,:,0], cmap="gray")
    ax[0, 0].set_title("Car Y Channel")
    ax[1, 0].imshow(car_ycrcb[:,:,1], cmap="gray")
    ax[1, 0].set_title("Car Cr Channel")
    ax[2, 0].imshow(car_ycrcb[:,:,2], cmap="gray")
    ax[2, 0].set_title("Car Cb Channel")

    not_car_ycrcb = cv2.cvtColor(not_car, cv2.COLOR_RGB2YCrCb)

    ax[0, 2].imshow(not_car_ycrcb[:,:,0], cmap="gray")
    ax[0, 2].set_title("Not Car Y Channel")
    ax[1, 2].imshow(not_car_ycrcb[:,:,1], cmap="gray")
    ax[1, 2].set_title("Not Car Cr Channel")
    ax[2, 2].imshow(not_car_ycrcb[:,:,2], cmap="gray")
    ax[2, 2].set_title("Not Car Cb Channel")

    _, car_hog_y = compute_hog(car_ycrcb[:,:,0], vis=True)
    _, car_hog_cr = compute_hog(car_ycrcb[:,:,1], vis=True)
    _, car_hog_cb = compute_hog(car_ycrcb[:,:,2], vis=True)

    ax[0, 1].imshow(car_hog_y, cmap="gray")
    ax[0, 1].set_title("Car HOG Y Channel")
    ax[1, 1].imshow(car_hog_cr, cmap="gray")
    ax[1, 1].set_title("Car HOG Cr Channel")
    ax[2, 1].imshow(car_hog_cb, cmap="gray")
    ax[2, 1].set_title("Car HOG Cb Channel")

    _, not_car_hog_y = compute_hog(not_car_ycrcb[:,:,0], vis=True)
    _, not_car_hog_cr = compute_hog(not_car_ycrcb[:,:,1], vis=True)
    _, not_car_hog_cb = compute_hog(not_car_ycrcb[:,:,2], vis=True)

    ax[0, 3].imshow(not_car_hog_y, cmap="gray")
    ax[0, 3].set_title("Not Car HOG Y Channel")
    ax[1, 3].imshow(not_car_hog_cr, cmap="gray")
    ax[1, 3].set_title("Not Car HOG Cr Channel")
    ax[2, 3].imshow(not_car_hog_cb, cmap="gray")
    ax[2, 3].set_title("Not Car HOG Cb Channel")

    plt.show()

    classifier, normalizer = load_classifier()

    test1 = cv2.imread("test_images/test1.jpg")
    test1 = cv2.cvtColor(test1, cv2.COLOR_BGR2RGB)
    windows = all_windows(test1)
    boxes = search_windows(test1, windows, classifier, normalizer)
    test1 = draw_boxes(test1, boxes, color=(255, 0, 0), thick=2)

    test2 = cv2.imread("test_images/test2.jpg")
    test2 = cv2.cvtColor(test2, cv2.COLOR_BGR2RGB)
    boxes = search_windows(test2, windows, classifier, normalizer)
    test2 = draw_boxes(test2, boxes, color=(255, 0, 0), thick=2)

    test3 = cv2.imread("test_images/test3.jpg")
    test3 = cv2.cvtColor(test3, cv2.COLOR_BGR2RGB)
    boxes = search_windows(test3, windows, classifier, normalizer)
    test3 = draw_boxes(test3, boxes, color=(255, 0, 0), thick=2)

    test4 = cv2.imread("test_images/test4.jpg")
    test4 = cv2.cvtColor(test4, cv2.COLOR_BGR2RGB)
    boxes = search_windows(test4, windows, classifier, normalizer)
    test4 = draw_boxes(test4, boxes, color=(255, 0, 0), thick=2)

    fig, ax = plt.subplots(2, 2, sharex="col", sharey="row")
    ax[0, 0].imshow(test1)
    ax[0, 0].set_title("test1.jpg")
    ax[0, 1].imshow(test2)
    ax[0, 1].set_title("test2.jpg")
    ax[1, 0].imshow(test3)
    ax[1, 0].set_title("test3.jpg")
    ax[1, 1].imshow(test4)
    ax[1, 1].set_title("test4.jpg")

    plt.show()

if __name__ == "__main__":
    output_plots()
