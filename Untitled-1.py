
import cv2
import numpy as np
import requests
import os

config_path = r"C:\Users\akshi\Object Detection\yolov3.cfg"
weights_path = r"C:\Users\akshi\Object Detection\yolov3.weights"
coco_names_path = r"C:\Users\akshi\Object Detection\coco.names"
image_path = r"C:\Users\akshi\Object Detection\Initial.jpg"
# Define the Django endpoint URL
endpoint_url = "http://127.0.0.1:8000/seatmap/"

previous_chair1 = False
previous_chair2 = False


# Load the pre-trained YOLO model
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Load the COCO class labels
with open(coco_names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load the initial image of unoccupied seats
initial_image = cv2.imread(image_path)

# Convert the initial image to blob for object detection
blob = cv2.dnn.blobFromImage(initial_image, 1/255, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Perform forward pass to get the initial chair detections
output_layers_names = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(output_layers_names)

# Get the initial chair bounding boxes and confidences
chairs = []
confidences = []
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if class_id == 0 and confidence > 0.5:  # Class ID 0 corresponds to "chair"
            center_x = int(detection[0] * initial_image.shape[1])
            center_y = int(detection[1] * initial_image.shape[0])
            width = int(detection[2] * initial_image.shape[1])
            height = int(detection[3] * initial_image.shape[0])
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)
            chairs.append([x, y, width, height])
            confidences.append(float(confidence))

# Define the regions of interest (ROIs) based on the chair bounding boxes
if len(chairs) > 0:
    roi1_x, roi1_y, roi1_width, roi1_height = chairs[0]
    roi2_x, roi2_y, roi2_width, roi2_height = chairs[0] + np.array([roi1_width + 10, 0, 0, 0])
    roi1 = (roi1_x, roi1_y, roi1_width, roi1_height)
    roi2 = (roi2_x, roi2_y, roi2_width, roi2_height)
else:
    print("No chair detected in the initial image. Drawing default ROIs.")
    # Set default ROIs if no chairs are detected
    roi1 = (100, 100, 200, 200)
    roi2 = (350, 100, 200, 200)

# Initialize the video capture from the camera
camera = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = camera.read()

    if not ret:
        break

    # Extract the ROIs from the frame
    roi1_frame = frame[roi1[1]:roi1[1] + roi1[3], roi1[0]:roi1[0] + roi1[2]]
    roi2_frame = frame[roi2[1]:roi2[1] + roi2[3], roi2[0]:roi2[0] + roi2[2]]

    # Convert the ROIs to blobs for object detection
    blob1 = cv2.dnn.blobFromImage(roi1_frame, 1/255, (416, 416), swapRB=True, crop=False)
    blob2 = cv2.dnn.blobFromImage(roi2_frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob1)
    layer_outputs1 = net.forward(output_layers_names)
    net.setInput(blob2)
    layer_outputs2 = net.forward(output_layers_names)

    # Reset the person presence status for the current frame
    person_present1 = False
    person_present2 = False

    # Iterate over the layer outputs and detect person presence in ROI 1
    for output in layer_outputs1:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:  # Class ID 0 corresponds to "person"
                person_present1 = True
                break

    # Iterate over the layer outputs and detect person presence in ROI 2
    for output in layer_outputs2:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:  # Class ID 0 corresponds to "person"
                person_present2 = True
                break

    # Draw bounding boxes on the ROIs
    color1 = (0, 255, 0) if person_present1 else (0, 0, 255)
    color2 = (0, 255, 0) if person_present2 else (0, 0, 255)
    cv2.rectangle(frame, (roi1[0], roi1[1]), (roi1[0] + roi1[2], roi1[1] + roi1[3]), color1, 2)
    cv2.rectangle(frame, (roi2[0], roi2[1]), (roi2[0] + roi2[2], roi2[1] + roi2[3]), color2, 2)
    
    # Determine the chair occupancy status
    # chair1 = person_present1
    # chair2 = person_present2
    chair1 = person_present1
    chair2 = person_present2
    
    # print(chair1)
    # print(chair2)
    
    # Prepare the data to send to the Django endpoint
    data = {
        'chair1': chair1,
        'chair2': chair2
    }

    print(data)
    
    if (previous_chair1 != chair1 or previous_chair2 != chair2):
    
        # Send a POST request to the Django endpoint
        header = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        response = requests.post(endpoint_url, json=data,headers=header)
        print(response)

        # Check the response status
        if response.status_code == 200:
            print("Data sent successfully")
        else:
            print("Failed to send data")
    
    previous_chair1 = chair1
    previous_chair2 = chair2

    # Display the resulting frame
    cv2.imshow('Chair Occupancy Detection', frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
camera.release()
cv2.destroyAllWindows()

