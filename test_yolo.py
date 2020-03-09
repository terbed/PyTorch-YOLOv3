from models import *
from utils.utils import *
import torch
import numpy as np
import cv2
tr = torch


def load_input(path, device):
    img = cv2.imread(path)
    inp = cv2.resize(img, (416, 416), interpolation=cv2.INTER_LINEAR)
    inp = cv2.cvtColor(inp, cv2.COLOR_RGB2BGR)

    inp = np.moveaxis(inp, 2, 0)    # C x H x W
    inp = tr.from_numpy(inp.astype(np.float32)).unsqueeze(0).to(device)

    return img, inp


device = tr.device('cuda') if tr.cuda.is_available() else tr.device('cpu')

model_def = 'config/yolov3-custom.cfg'
class_path = 'data/custom/classes.names'
weight_path = 'weights/yolov3_ckpt_65.pth'
img_path = '/media/terbe/sztaki/DATA/BabyCropper/data/test_baby/'
img_name = '000028.png'

# parameters
conf_thres = 0.5
nms_thres = 0.5

# Load model
model = Darknet(model_def).to(device)
model.load_state_dict(torch.load(weight_path))
model.eval()

# Load classes
classes = load_classes(class_path)  # Extracts class labels from file

# Load image
img, inp = load_input(img_path+img_name, device)
# print(inp.shape)


with tr.set_grad_enabled(False):
    outputs = model(inp)
print('Output calculated! ')

outputs = non_max_suppression(outputs, conf_thres, nms_thres)[0]
print(img.shape[:2])
detections = rescale_boxes(outputs, 416, img.shape[:2])
unique_labels = detections[:, -1].cpu().unique()
n_cls_preds = len(unique_labels)

for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

    box_w = x2 - x1
    box_h = y2 - y1

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 500, 500)
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))

cv2.imshow('frame', img)

while cv2.waitKey(1) != 13:
    pass
