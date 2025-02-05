import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLO model
model = YOLO('best.pt')

def detect_objects(imagep):
    image = cv2.imread(imagep)
    results = model(image)
    plants, fruits = [], []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])  # 0: Red, 1: Yellow, 2: Purple, 3: Plant

            if class_id == 3:  # Plant
                plants.append((x1, y1, x2, y2))
            else:  # Fruit
                fruits.append((x1, y1, x2, y2, class_id))

    return plants, fruits

def assign_normalized(plants, fruits):
    transformed_fruits = {m: [] for m in range(len(plants))}

    for fruit in fruits:
        x1, y1, x2, y2, cls = fruit
        centre_fruit = ((x1 + x2) / 2, (y1 + y2) / 2)
        best_plant, min_distance = None, float("inf")

        for m, (px1, py1, px2, py2) in enumerate(plants):
            centre_plant = ((px1 + px2) / 2, (py1 + py2) / 2)
            distance = np.linalg.norm(np.array(centre_fruit) - np.array(centre_plant))

            if distance < min_distance:
                min_distance = distance
                best_plant = m

        if best_plant is not None:
            px1, py1, px2, py2 = plants[best_plant]
            center_plant = ((px1 + px2) / 2, (py1 + py2) / 2)

            # Normalize coordinates relative to plant size
            plant_width, plant_height = max(px2 - px1, 1), max(py2 - py1, 1)

            normalized_x = (centre_fruit[0] - center_plant[0]) / plant_width
            normalized_y = (centre_fruit[1] - center_plant[1]) / plant_height

            transformed_fruits[best_plant].append([normalized_x, normalized_y, cls])

    return transformed_fruits

def remove_duplicate_fruits(normalized_front, normalized_rear,threshold=0.04):
    unique_fruits = {m: [] for m in normalized_front.keys()}

    for plant_id in normalized_front:
        front_fruits = normalized_front[plant_id]
        back_fruits = normalized_rear.get(plant_id, [])
        matched_indices = set()

        # Step 1: Process front view fruits
        for f1 in front_fruits:
            x1, y1, cls1 = f1

            for i, f2 in enumerate(back_fruits):
                x2, y2, cls2 = f2

                if cls1 == cls2:  # Same fruit color
                    distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

                    if distance < threshold:  # If fruit positions are close, consider duplicate
                        matched_indices.add(i)
                        break

            # If not duplicate, add it to the unique list
            unique_fruits[plant_id].append(f1)

        for i, f2 in enumerate(back_fruits):
            if i not in matched_indices:  # Only add unmatched fruits from back view
                unique_fruits[plant_id].append(f2)
    unique_fruits_count = {m: len(unique_fruits[m]) for m in normalized_front.keys()}

    return unique_fruits, unique_fruits_count



# Load and process images
front_plants, front_fruits = detect_objects("images_close_img_21_2.jpg")

# Mirror rear image
rear = cv2.imread("images_close_img_21_6.jpg")
mirrored_rear = cv2.flip(rear, 1)
cv2.imwrite("mirrored_image.jpg", mirrored_rear)

rear_plants, rear_fruits = detect_objects("mirrored_image.jpg")

# Assign fruits to plants with normalized coordinates
normalized_front= assign_normalized(front_plants, front_fruits)
normalized_rear= assign_normalized(rear_plants, rear_fruits)

# Remove duplicate fruits
uniq,uniq_count = remove_duplicate_fruits(normalized_front, normalized_rear)

print(uniq)
print(uniq_count)
