import torch
import cv2
import pandas as pd
from torchvision.ops import box_iou, nms

model = torch.hub.load('yolov5', 'yolov5x', source='local')

cap = cv2.VideoCapture('video-1.mp4')

vehicle_counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0, 'lorry': 0}

prev_boxes = None  # Track previous frame's bounding boxes
tracked_vehicles = {}  # Dictionary to store tracked vehicles

confidence_threshold = 0.6  # Minimum confidence threshold for counting vehicles
disappeared_threshold = 10  # Number of consecutive frames a vehicle should be missing to be considered disappeared

frame_count = 0

# Read the first frame to initialize prev_boxes
_, img = cap.read()
result = model(img)
df = result.pandas().xyxy[0]
prev_boxes = torch.tensor(df[['xmin', 'ymin', 'xmax', 'ymax']].values, dtype=torch.float32)

while True:
    img = cap.read()[1]
    if img is None:
        break
    result = model(img)
    df = result.pandas().xyxy[0]

    scores = df['confidence'].tolist()
    boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values

    # Convert boxes and scores to tensors
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores)

    # Calculate IoU between current and previous frame's bounding boxes
    ious = box_iou(boxes_tensor, prev_boxes)

    # Filter out boxes with high overlap with previous frame's boxes
    overlap_indices = torch.any(ious > 0.5, dim=1)
    boxes_tensor = boxes_tensor[~overlap_indices]
    scores_tensor = scores_tensor[~overlap_indices]

    # Apply non-maximum suppression to remove redundant bounding boxes
    keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
    df = df.iloc[keep]


    prev_boxes = boxes_tensor.clone()


    new_tracked_vehicles = {}

    for ind in df.index:
        x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
        x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])
        label = df['name'][ind]
        conf = df['confidence'][ind]

        if conf > confidence_threshold:
            # Check if the bounding box overlaps or is close to any existing tracked vehicle
            matched_vehicle_id = None
            for vehicle_id, (vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, vehicle_confidence) in tracked_vehicles.items():
                if x1 < vehicle_x2 and x2 > vehicle_x1 and y1 < vehicle_y2 and y2 > vehicle_y1:
                    matched_vehicle_id = vehicle_id
                    break

            if matched_vehicle_id is None:
                # Create a new tracked vehicle and assign an ID
                matched_vehicle_id = len(tracked_vehicles) + 1

            if matched_vehicle_id in tracked_vehicles:
                tracked_vehicles[matched_vehicle_id] = (x1, y1, x2, y2, vehicle_confidence + 1)
            else:
                tracked_vehicles[matched_vehicle_id] = (x1, y1, x2, y2, 1)

            # Increment the vehicle count if it has been confidently detected over several frames
            if tracked_vehicles[matched_vehicle_id][4] >= disappeared_threshold:
                if label in vehicle_counts:
                    vehicle_counts[label] += 1

            new_tracked_vehicles[matched_vehicle_id] = tracked_vehicles[matched_vehicle_id]

    # Remove disappeared vehicles
    disappeared_vehicles = []
    for vehicle_id, (vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, vehicle_confidence) in tracked_vehicles.items():
        if vehicle_id not in new_tracked_vehicles:
            if frame_count - disappeared_threshold > 0:
                disappeared_vehicles.append(vehicle_id)

    for disappeared_vehicle_id in disappeared_vehicles:
        del tracked_vehicles[disappeared_vehicle_id]

    # Draw bounding boxes for tracked vehicles
    for vehicle_id, (vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, vehicle_confidence) in tracked_vehicles.items():
        cv2.rectangle(img, (vehicle_x1, vehicle_y1), (vehicle_x2, vehicle_y2), (0, 255, 0), 2)

    # Draw vehicle count on the image
    for i, (vehicle_type, count) in enumerate(vehicle_counts.items()):
        text = vehicle_type + ' Count: ' + str(count)
        cv2.putText(img, text, (50, 50 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Video', img)
    k = cv2.waitKey(10)
    if k == ord('q'):
        break

    frame_count += 1

df_counts = pd.DataFrame({'Vehicle': list(vehicle_counts.keys()), 'Count': list(vehicle_counts.values())})
df_counts.to_csv('vehicle_counts.csv', index=False)

cv2.destroyAllWindows()
