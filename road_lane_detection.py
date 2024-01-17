import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(255, 0, 0), thickness=5):
    for line in lines:
        line = line.reshape(-1)  # Flatten the array if it has more than one dimension
        x1, y1, x2, y2 = line[:4]
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def extrapolate_lines(lines, height):
    left_lines, right_lines = [], []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

    left_avg = np.average(left_lines, axis=0)
    right_avg = np.average(right_lines, axis=0)

    left_line = extrapolate_line(left_avg, height)
    right_line = extrapolate_line(right_avg, height)

    return [left_line, right_line]

def extrapolate_line(line, y):
    x1, y1, x2, y2 = line.squeeze()
    slope = (y2 - y1) / (x2 - x1)
    x_intercept = x1 - (y1 / slope)
    x = int((y - x_intercept) / slope)
    return np.array([[x, int(y), int((height - x_intercept) / slope), height]], dtype=np.int32)

# Function to process each frame of the video
def process_frame(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define the region of interest (ROI)
    roi_vertices = np.array([[(0, height), (width // 2, height // 2), (width, height)]], dtype=np.int32)
    roi_edges = region_of_interest(edges, roi_vertices)

    # Apply Hough Transform to detect lines within the ROI
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

    # Create a blank image to draw lines on
    line_img = np.zeros_like(frame, dtype=np.uint8)

    # Extrapolate and draw two solid lines representing the left and right lane markings
    extrapolated_lines = extrapolate_lines(lines, height)

    # Convert each line array to the appropriate data type
    extrapolated_lines = [line.astype(int) for line in extrapolated_lines]

    # Draw the extrapolated lines on the frame
    draw_lines(line_img, extrapolated_lines)

    # Color the lane area between the two extrapolated lines
    lane_area = np.zeros_like(frame)
    cv2.fillPoly(lane_area, extrapolated_lines, (0, 255, 0))

    # Combine the lane area with the original frame
    result = cv2.addWeighted(frame, 0.8, lane_area, 0.5, 0)

    return result

# Load the sample road image
image = cv2.imread('road_detection.jpg')

# Define the region of interest (ROI)
height, width = image.shape[0], image.shape[1]
roi_vertices = np.array([[(width * 0.1, height), (width * 0.45, height * 0.6), (width * 0.55, height * 0.6), (width * 0.9, height)]], dtype=np.int32)
roi_image = region_of_interest(image, roi_vertices)

# Convert the ROI image to grayscale
gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

# Save the grayscale image
cv2.imwrite('grayscale_image.jpg', gray_roi)

# Apply Gaussian Blur to reduce noise and improve edge detection
blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

# Save the blurred image
cv2.imwrite('blurred_image.jpg', blurred_roi)

# Apply Canny Edge Detection
edges_roi = cv2.Canny(blurred_roi, 50, 150)

# Save the edges image
cv2.imwrite('edges_image.jpg', edges_roi)

# Apply Hough Transform to detect lines within the ROI
lines_roi = cv2.HoughLinesP(edges_roi, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

# Create a blank image to draw lines on
line_img = np.zeros_like(roi_image, dtype=np.uint8)

# Extrapolate and draw two solid lines representing the left and right lane markings
extrapolated_lines = extrapolate_lines(lines_roi, height * 0.6)

# Convert each line array to the appropriate data type
extrapolated_lines = [line.astype(int) for line in extrapolated_lines]

# Draw the extrapolated lines on the image
draw_lines(line_img, extrapolated_lines)

# Color the lane area between the two extrapolated lines
lane_area = np.zeros_like(roi_image)
cv2.fillPoly(lane_area, extrapolated_lines, (0, 255, 0))

# Combine the lane area with the original ROI image
result_roi = cv2.addWeighted(roi_image, 0.8, lane_area, 0.5, 0)

# Display and save the result
cv2.imshow('Lane Detection in ROI', result_roi)
cv2.imwrite('output_lane_detection_roi.jpg', result_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load the video file
video_path = 'road_detection.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(5)

# Create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output_lane_detection_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process each frame in the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    processed_frame = process_frame(frame)

    # Display the processed frame
    cv2.imshow('Lane Detection in Video', processed_frame)
    
    # Write the processed frame to the output video
    out.write(processed_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()
