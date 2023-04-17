import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def calc_hist(img):
    # Calculate the histogram of a grayscale image
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist.flatten()

def hist_diff(hist1, hist2):
    # Calculate the absolute difference between two histograms
    diff = np.sum(np.abs(hist1 - hist2))
    return diff

def detect_shots_from_video(filename):
    # Open the video file
    cap = cv2.VideoCapture(filename)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize variables for shot detection
    prev_frame = None
    prev_hist = None
    threshold = 40000  # Adjust this parameter to adjust the sensitivity of the shot detector

    # array for tracking changes
    changes = []

    # Loop over the frames of the video
    for i in range(frame_count):
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the histogram of the current frame
        curr_hist = calc_hist(gray)

        if prev_frame is not None:
            # Calculate the absolute difference between the current and previous histograms
            diff = hist_diff(curr_hist, prev_hist)

            # Check if the absolute difference exceeds the threshold
            if diff > threshold:
                # A shot has been detected
                print("Shot detected at time {:.2f} seconds".format(i / fps))

                changes.append(1)
            
            else:
                changes.append(0)

        # Update the previous frame and histogram
        prev_frame = gray
        prev_hist = curr_hist

    # Release the video capture object and close the windows
    cap.release()

    # plot the array using matplotlib
    plt.plot(changes)

    # Add labels and title
    plt.xlabel("Frames")
    plt.ylabel("Change")
    plt.title("Change detection")

    plt.show()

def detect_shots_from_list(image_list):
    """
    Input: image_list of images of type PIL
    Output: indices where change happens
    """
    image_num = len(image_list)

    # Initialize variables for shot detection
    prev_frame = None
    prev_hist = None
    threshold = 140000  # Adjust this parameter to adjust the sensitivity of the shot detector

    # array for tracking changes
    changes = []
    ind_of_change = []
    diffs = []

    # Loop over the images
    for i in range(image_num):
        # Read the next frame
        image = image_list[i]

        # Convert the image from PIL to grayscale OpenCv image
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Calculate the histogram of the current frame
        curr_hist = calc_hist(gray)

        if prev_frame is not None:
            # Calculate the absolute difference between the current and previous histograms
            diff = hist_diff(curr_hist, prev_hist)
            diffs.append(diff)

            # Check if the absolute difference exceeds the threshold
            if diff > threshold:
                # A shot has been detected
                print("Shot detected at index {:.2f}".format(i))
                changes.append(1)
                ind_of_change.append(i)
            
            else:
                changes.append(0)

        else:
            # for the 0th frame there is no prev frame
            ind_of_change.append(i) # mark change
            changes.append(0)
            diffs.append(0)

        # Update the previous frame and histogram
        prev_frame = gray
        prev_hist = curr_hist

    # plot the array using matplotlib
    fig, ax = plt.subplots(figsize = (20, 5))
    ax.plot(range(image_num), changes)

    # normalize the differences
    # diffs = (diffs - np.min(diffs)) / (np.max(diffs) - np.min(diffs))
    ax.plot(range(image_num), diffs)

    # Add labels and title
    plt.xlabel("Frames")
    plt.ylabel("Change")
    plt.title("Change detection")

    # add a vertical line at an index where a change occurs
    for ind in ind_of_change:
        ax.axvline(x=ind, color='r')
        ax.text(ind + 0.1, 0, f'i={ind}', rotation=90)

    plt.show()

    return ind_of_change

def detect_shots_from_list_label(image_list):
    """
    Input: image_list of images of type PIL and its label: (PIL Image, 0)
    Output: a modified tuple where the labels of each image are set based on belonging to a set of consequent frames with no change
    """
    labeled_images_list = []
    image_num = len(image_list)

    # Initialize variables for shot detection
    prev_frame = None
    prev_hist = None
    threshold = 140000  # Adjust this parameter to adjust the sensitivity of the shot detector
    prev_class = 0

    # array for tracking changes
    changes = []
    ind_of_change = []
    diffs = []
    class_labels = [prev_class]

    # Loop over the images
    for i in range(image_num):
        # Read the next frame
        image = image_list[i][0]

        # Convert the image from PIL to grayscale OpenCv image
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Calculate the histogram of the current frame
        curr_hist = calc_hist(gray)

        if prev_frame is not None:
            # Calculate the absolute difference between the current and previous histograms
            diff = hist_diff(curr_hist, prev_hist)
            diffs.append(diff)

            # Check if the absolute difference exceeds the threshold
            if diff > threshold:
                # A shot has been detected
                print("Shot detected at index {:.2f}".format(i))
                changes.append(1)
                ind_of_change.append(i)
                curr_class = prev_class + 1
                labeled_images_list.append((image, curr_class))
                # Add a class label to the list of class labels
                class_labels.append(curr_class)

            else:
                changes.append(0)
                labeled_images_list.append((image, prev_class))

        else:
            # for the 0th frame there is no prev frame
            ind_of_change.append(i) # mark change
            changes.append(0)
            diffs.append(0)
            curr_class = 0
            labeled_images_list.append((image, curr_class))


        # Update the previous frame and histogram
        prev_frame = gray
        prev_hist = curr_hist
        prev_class = curr_class

    print(f'Num of changes: {len(ind_of_change)}')
    print(f'Num of classes: {curr_class+1}')


    # plot the array using matplotlib
    fig, ax = plt.subplots(figsize = (20, 5))
    ax.plot(range(image_num), changes)

    # normalize the differences
    diffs = (diffs - np.min(diffs)) / (np.max(diffs) - np.min(diffs))
    ax.plot(range(image_num), diffs)

    # Add labels and title
    plt.xlabel("Frames")
    plt.ylabel("Change")
    plt.title("Change detection")

    # add a vertical line at an index where a change occurs
    for ind in ind_of_change:
        ax.axvline(x=ind, color='r')
        ax.text(ind + 0.1, 0, f'i={ind}', rotation=90)

    plt.show()

    for i, (im, label) in enumerate(labeled_images_list):
        print(f'index {i}, class {label}')

    return labeled_images_list, class_labels

if __name__ == "__main__":
    path = sys.argv[1]

    if os.path.isfile(path):
        detect_shots_from_video(path)