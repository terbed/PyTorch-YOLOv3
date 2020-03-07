from models import *
from utils.utils import *
import torch
import numpy as np
import cv2
tr = torch


def load_input(path, device):
    img = cv2.imread(path)
    img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = np.moveaxis(img, 2, 0)    # C x H x W
    img = tr.from_numpy(img.astype(np.float32)).unsqueeze(0).to(device)

    return img


device = tr.device('cuda') if tr.cuda.is_available() else tr.device('cpu')

model_def = 'config/yolov3-custom.cfg'
weight_path = 'weights/yolov3_ckpt_99.pth'
img_path = '/media/terbe/sztaki/DATA/BabyCropper/data/illustration_selection/'
img_name = '2020y2m16d_9h32m_000065.png'

# parameters
conf_thres = 0.001
nms_thres = 0.5

# Initiate model
model = Darknet(model_def).to(device)
model.load_state_dict(torch.load(weight_path))

model.eval()

inp = load_input(img_path+img_name, device)
print(inp.shape)

with tr.set_grad_enabled(False):
    outputs = model(inp)
print('Output calculated! ')

outputs = non_max_suppression(outputs)
print(outputs)
print(outputs.size())

pred_boxes = outputs[:, :4]
pred_scores = outputs[:, 4]

print(pred_boxes)