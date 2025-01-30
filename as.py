import cv2
from ultralytics import YOLO
import numpy as np

model=YOLO('best.pt')

front_image=cv2.imread("images_close_img_21_2.jpg")
back_image=cv2.imread("images_close_img_3_0_back.jpg")


f_val=model(front_image)
b_val=model(back_image)

print(f_val)
print(b_val)
for r in f_val:
    print(r.boxes)

'''def fbox(val):
    fruit_boxes=[]
    for r in val:
        for box in r.boxes:
            x1,y1,x2,y2=box.xyxy[0].tolist()
            fruit_boxes.append([x1,x2,y1,y2])
    return fruit_boxes

fruit_boxes_front=fbox(f_val)
fruit_boxes_back=fbox(b_val)

print(fruit_boxes_front,fruit_boxes_back)'''



