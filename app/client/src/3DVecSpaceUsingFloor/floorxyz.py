import cv2
import numpy as np

def find_potential_lines(image):
  # Convert to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply Canny edge detection
  edges = cv2.Canny(gray, 50, 150, apertureSize=3)

  # Find lines using Hough Lines Transform
  lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/360, threshold=100, minLineLength=20, maxLineGap=40)

  # Potential lines for horizon, wall, and floor
  potential_horizon = None
  potential_wall = None
  potential_floor = None

  # Loop through detected lines
  for line in lines:
    x1, y1, x2, y2 = line[0]

    # Check for horizontal lines (potential horizon)
    if abs(y2 - y1) < 10:
      if potential_horizon is None or abs(y1 - image.shape[0] / 2) < abs(potential_horizon[1] - image.shape[0] / 2):
        potential_horizon = line[0]

    # Check for lines with large vertical change (potential wall)
    elif abs(x2 - x1) < 20 and abs(y2 - y1) > 100:
      potential_wall = line[0]

    # Check for lines close to the bottom (potential floor)
    elif y1 > image.shape[0] * 3/4 and y2 > image.shape[0] * 3/4:
      potential_floor = line[0]

  return potential_horizon, potential_wall, potential_floor

# Load image
image = cv2.imread("nb/ref.jpg")

# Find potential lines
potential_horizon, potential_wall, potential_floor = find_potential_lines(image.copy())

# Draw lines on a copy of the image (optional)
if potential_horizon is not None:
  cv2.line(image, (potential_horizon[0], potential_horizon[1]), (potential_horizon[2], potential_horizon[3]), (0, 255, 0), 2)  # Green for horizon
if potential_wall is not None:
  cv2.line(image, (potential_wall[0], potential_wall[1]), (potential_wall[2], potential_wall[3]), (0, 0, 255), 2)  # Red for wall
if potential_floor is not None:
  cv2.line(image, (potential_floor[0], potential_floor[1]), (potential_floor[2], potential_floor[3]), (255, 0, 0), 2)  # Blue for floor

# Display the image with potential lines
cv2.imshow("Image with Lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
