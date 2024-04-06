import cv2
import numpy as np

# Load and convert image to grayscale
image = cv2.imread('nb/ref.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply Hough Transform for line detection
minLineLength = 10
maxLineGap = 10
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength, maxLineGap)

# Draw detected lines on the original image
angles = []
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        angle = 0 - np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)
        cv2.putText(image, f"Angle: {angle:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# find top 3 angles tahat appear most frequently, do not repeat
angles = np.array(angles)
angles = np.round(angles, 1)
unique, counts = np.unique(angles, return_counts=True)
angle_counts = dict(zip(unique, counts))
angle_counts = dict(sorted(angle_counts.items(), key=lambda item: item[1], reverse=True))
top_angles = list(angle_counts.keys())[:3]
print(top_angles)

# find unique angles from all angles with difference threshold of 3 degrees
unique_angles = []
for angle in angles:
    if len(unique_angles) == 0:
        unique_angles.append(angle)
    else:
        if all(abs(angle - u) > 3 for u in unique_angles):
            unique_angles.append(angle)
print('U3:',unique_angles)

# Show the image with detected lines
cv2.imshow('Detected Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
