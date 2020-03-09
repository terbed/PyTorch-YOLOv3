import numpy as np
import cv2
import os

path_img = '/media/terbe/sztaki/DATA/BabyCropper/data/images/'
out_path = '/media/terbe/sztaki/DATA/BabyCropper/data/images128/'
os.mkdir(out_path)

# Create sets for names
img_names = []

# Fill image set with names
with os.scandir(path_img) as entries:
    for entry in entries:
        current_name = entry.name
        print(current_name)
        img_names.append(current_name)

for name in img_names:
    current_img = cv2.imread(path_img + name)
    current_img = cv2.resize(current_img, (128, 128), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path+name, current_img)
    print(name)
