import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip, ImageSequenceClip
from utils import load_classifier, all_windows, search_windows, heatmap_image, filter_heatmaps, draw_labels
import sys

num_heatmaps = 7
heatmaps = []

def detect_cars(img, windows, classifier, normalizer):
    # Scan frame for cars
    boxes = search_windows(img, windows, classifier, normalizer)

    # Build heatmap for this frame
    heatmap = heatmap_image(img, boxes)

    # Initialize buffer of heatmaps
    if len(heatmaps) == 0:
        for i in range(num_heatmaps):
            heatmaps.append(heatmap.copy())

    # Cycle the heatmap buffer
    heatmaps.pop(0)
    heatmaps.append(heatmap)

    # Combine/filter the heatmap buffer
    labels = filter_heatmaps(heatmaps, threshold=4)

    return draw_labels(img, labels, color=(255, 0, 0), thick=2)

# Make sure we got a movie file
if len(sys.argv) != 2:
    print("Usage: python detect.py <movie file>")
    sys.exit(1)
movie_file = sys.argv[1]

# Load the classifier (must train first)
classifier, normalizer = load_classifier()
print("Loaded classifier.")

# Initialize detection windows from test image.
img = cv2.imread("test_images/test7.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
windows = all_windows(img)
print("Using {} detection windows.".format(len(windows)))

# Process each frame in the movie file
print("Processing {} -> output.mp4".format(movie_file))
in_clip = VideoFileClip(movie_file)
out_frames = []
frame_count = 0
for in_frame in in_clip.iter_frames():
    frame_count += 1
    img = detect_cars(in_frame, windows, classifier, normalizer)
    out_frames.append(img)
    print("Processed frame {}".format(frame_count))
out_clip = ImageSequenceClip(out_frames, fps=in_clip.fps)
out_clip.write_videofile("output.mp4")
