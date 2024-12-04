import numpy as np
import cv2 as cv

# Load and preprocess the template image
img1 = cv.imread('./botella2.jpeg', cv.IMREAD_GRAYSCALE)
target_width = 400
aspect_ratio = img1.shape[1] / img1.shape[0]
target_height = int(target_width / aspect_ratio)
img1 = cv.resize(img1, (target_width, target_height), interpolation=cv.INTER_AREA)

# Open the webcam
cam = cv.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Known object dimensions and calibration distance
W_real = 25  # in centimeters
D_known = 10  # in centimeters
focal_length = None

# Initialize SIFT detector and FLANN matcher
sift = cv.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

# Compute keypoints and descriptors for the template image once
kp1, des1 = sift.detectAndCompute(img1, None)

def filter_matches(matches, ratio=0.5):
    """Filter matches using Lowe's ratio test."""
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

def calculate_object_width(good_matches, kp2):
    """Estimate object width in pixels based on matched keypoints."""
    matched_points = [kp2[m.trainIdx].pt for m in good_matches]
    if matched_points:
        x_coords = [pt[0] for pt in matched_points]
        return max(x_coords) - min(x_coords)
    return None

while True:
    status, photo = cam.read()
    if not status:
        print("Error: Could not read frame from webcam.")
        break

    # Convert frame to grayscale
    gray_photo = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)

    # Compute keypoints and descriptors for the current frame
    kp2, des2 = sift.detectAndCompute(gray_photo, None)

    if des1 is not None and des2 is not None:
        # Match descriptors and filter them
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = filter_matches(matches, ratio=0.5)

        # Calculate object width in pixels
        W_pixel = calculate_object_width(good_matches, kp2)

        # Calibrate focal length if not done yet
        if W_pixel and not focal_length:
            focal_length = (W_pixel * D_known) / W_real
            print(f"Calibrated focal length: {focal_length:.2f}")

        # Calculate distance to the object
        if W_pixel and focal_length:
            distance = (W_real * focal_length) / W_pixel
            print(f"Estimated Distance: {distance:.2f} cm")
        else:
            print("Object not detected.")

        # Visualize the good matches
        img3 = cv.drawMatches(img1, kp1, gray_photo, kp2, good_matches, None,
                              matchColor=(0, 255, 0),
                              singlePointColor=(255, 0, 0),
                              flags=cv.DrawMatchesFlags_DEFAULT)
        cv.imshow("Matches", img3)
    else:
        print("Descriptors not found.")

    # Press 'q' to quit
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv.destroyAllWindows()
