import cv2
import numpy as np
import csv

def detect_ball(frame, lower_bound, upper_bound):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

def get_region(point, regions):
    for i, region in enumerate(regions):
        if region[0][0] <= point[0] <= region[1][0] and region[0][1] <= point[1] <= region[1][1]:
            return i + 1
    return "out_of_bound"

# Create video capture object
cap = cv2.VideoCapture("C:/Users/yazhi/Desktop/CV_Builder_Series/projects/Input.mov")

# Get original video properties
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Original Video Dimensions: {original_width}x{original_height}")

# Define HSV bounds for ball detection (example bounds)
lower_bound = np.array([35, 40, 40])
upper_bound = np.array([85, 255, 255])

# Define regions in the 2x2 grid
regions = [
    [(0, 0), (original_width // 2, original_height // 2)],
    [(original_width // 2, 0), (original_width, original_height // 2)],
    [(0, original_height // 2), (original_width // 2, original_height)],
    [(original_width // 2, original_height // 2), (original_width, original_height)]
]

bounce_count = 0
prev_y = None
ball_moving_up = False

# CSV file to save bounce information
with open('bounce_info.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Bounce Count', 'Region', 'Timestamp', 'Frame Number'])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect the ball
        contour = detect_ball(frame, lower_bound, upper_bound)
        if contour is not None:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cX, cY), 10, (0, 255, 0), -1)

                # Get the current region of the ball
                region = get_region((cX, cY), regions)

                # Detect bounce based on vertical movement
                if prev_y is not None:
                    if cY < prev_y and not ball_moving_up:
                        bounce_count += 1
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        writer.writerow([bounce_count, region, timestamp, frame_number])
                        print(f"Bounce: {bounce_count}, Region: {region}, Time: {timestamp}, Frame: {frame_number}")
                        ball_moving_up = True
                    elif cY > prev_y:
                        ball_moving_up = False

                prev_y = cY

        # Display bounce count and time on video
        cv2.putText(frame, f"Count: {bounce_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Time: {cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame in full screen
        cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()