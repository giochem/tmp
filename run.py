from ultralytics import YOLO
import cv2

# Load object detection
detect_model = YOLO('gender_and_vehicle_detect.pt')
# print(detect_model)

# Run inference on an image
# https://docs.ultralytics.com/modes/predict/#working-with-results
numpy_img = cv2.imread("bus.jpg")
results = detect_model(numpy_img)  # results list

# View results
for r in results:
    print(r.names)  # A dictionary of class names.
    print(r.boxes)  # The boxes object containing the detection bounding boxes

# Load gender classify
# classify_model = YOLO('gender_classify.pt')
# numpy_img_2 = cv2.imread("male.jpg")
# classify_results = classify_model.predict(numpy_img_2)