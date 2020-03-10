from models import *
from utils.utils import *
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
tr = torch


def load_input(img_path, device):
    img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
    # Resize
    img = F.interpolate(img.unsqueeze(0), size=416, mode="nearest")
    img = img.type(tr.FloatTensor)
    img = img.to(device)

    return img


device = tr.device('cuda') if tr.cuda.is_available() else tr.device('cpu')

model_def = 'config/yolov3-custom.cfg'
class_path = 'data/custom/classes.names'
weight_path = 'weights/yolov3_ckpt_42.pth'
img_path = '/media/terbe/sztaki/DATA/BabyCropper/data/test_baby/'
img_name = '000028.png'

# parameters
conf_thres = 0.8
nms_thres = 0.4

# Load model
model = Darknet(model_def).to(device)
model.load_state_dict(torch.load(weight_path))
model.eval()

# Load classes
classes = load_classes(class_path)  # Extracts class labels from file

# Load image
inp = load_input(img_path+img_name, device)
img = cv2.imread(img_path+img_name)

with tr.set_grad_enabled(False):
    outputs = model(inp)
    outputs = non_max_suppression(outputs, conf_thres, nms_thres)[0]
print('Output calculated! ')

print(img.shape[:2])
detections = rescale_boxes(outputs, 416, img.shape[:2])
unique_labels = detections[:, -1].cpu().unique()
n_cls_preds = len(unique_labels)

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 500, 500)
for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

    box_w = x2 - x1
    box_h = y2 - y1

    print(x1, y1, x2, y2)

    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 55), 2)
    cv2.imshow('frame', img)
    while cv2.waitKey(1) != 13:
        pass


cv2.imshow('frame', img)
while cv2.waitKey(1) != 13:
    pass
