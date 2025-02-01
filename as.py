import cv2
from ultralytics import YOLO
import numpy as np

model=YOLO('best.pt')

def detect_objects(imagep):
    image = cv2.imread(imagep)
    results = model(image)

    plants = []
    fruits = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])  # 0: Red, 1: Yellow, 2: Purple, 3: Plant

            if class_id == 3:  # Plant
                plants.append((x1, y1, x2, y2))
            else:  # Fruit
                fruits.append((x1, y1, x2, y2, class_id))

    return plants, fruits

def assign_fruits_to_plants(plants, fruits):
    plant_fruit_count = {i: {0: 0, 1: 0, 2: 0} for i in range(len(plants))}#doing this to initialize each plant with each fruit set to 0

    for fruit in fruits:
        x1, y1, x2, y2, class_id = fruit
        fruit_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        best_match = None
        min_distance = float("inf")

        for i, (px1, py1, px2, py2) in enumerate(plants):
            plant_center = ((px1 + px2) // 2, (py1 + py2) // 2)
            distance = np.linalg.norm(np.array(fruit_center) - np.array(plant_center))

            if distance < min_distance:
                min_distance = distance
                best_match = i

        if best_match is not None:
            plant_fruit_count[best_match][class_id] += 1

    return plant_fruit_count

front_plants, front_fruits = detect_objects("images_close_img_7_4.jpg")
rear=cv2.imread("images_close_img_7_20.jpg")
flip=cv2.flip(rear,1)
cv2.imwrite("mirrored_image.jpg", flip)
rear_plants,rear_fruits = detect_objects("mirrored_image.jpg")


# Count fruits per plant
plant_fruit_counts1 = assign_fruits_to_plants(front_plants, front_fruits)
print(plant_fruit_counts1)
plant_fruit_counts2 = assign_fruits_to_plants(rear_plants, rear_fruits)
print(plant_fruit_counts2)

plant_fruit_counts={}
for i in range(3):
    plant_fruit_counts[i] = Counter(plant_fruit_counts1[i]) + Counter(plant_fruit_counts2[i])

# Display Results
for plant_id, fruit_count in plant_fruit_counts.items():
    print(f"Plant {plant_id}: Red={fruit_count[0]}, Yellow={fruit_count[1]}, Purple={fruit_count[2]}")




