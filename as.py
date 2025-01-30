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



