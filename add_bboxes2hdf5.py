import h5py
import argparse
import tqdm

from models import *
from utils.utils import *
from utils.datasets import *

import torch
from torch.utils.data import DataLoader
tr = torch


def get_baby_box(det, img_size):
    if det is not None:
        x_1 = y_1 = x_2 = y_2 = 0
        prev_conf = 0
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in det:
            # Crop baby
            if classes[int(cls_pred)] == 'baby':
                if prev_conf < cls_conf:
                    prev_conf = cls_conf
                    x_1, y_1, x_2, y_2 = round(x1.item()), round(y1.item()), round(x2.item()), round(y2.item())

        # check to be inside image size boundaries
        if y_2 > img_size:
            y_2 = img_size
        if x_2 > img_size:
            x_2 = img_size
        if y_1 < 0:
            y_1 = 0
        if x_1 < 0:
            x_1 = 0
        # check validity
        if y_2 - y_1 < 1 or x_2 - x_1 < 1:
            y_1 = x_1 = 0
            y_2 = x_2 = img_size

        return x_1, y_1, x_2, y_2
    else:
        print('NO OBJECT WAS FOUND!!!')
        return 0, 0, 128, 128   # return the full size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--hdf5_path", type=str, default="", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=128, help="size of image in hdf5")
    parser.add_argument("--interp_mode", type=str, default="nearest", help="area, bicubic, nearest")
    args = parser.parse_args()
    print(args)

    device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')

    # --------------------
    # Load model
    # -------------------
    model = Darknet(args.model_def).to(device)
    model.load_state_dict(torch.load(args.weights_path))
    model.eval()

    # Load classes
    classes = load_classes(args.class_path)  # Extracts class labels from file

    dataset = ListDatasetHDF5(args.hdf5_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_cpu, pin_memory=True)

    for batch_i, x in enumerate(tqdm.tqdm(dataloader, 'Detecting baby')):
        with tr.no_grad():
            x = x.to(device)
            outputs = model(x)
            outputs = non_max_suppression(outputs, args.conf_thres, args.nms_thres)

        detections = [rescale_boxes(output, 416, (args.img_size, args.img_size)) if output is not None else None for output in outputs]
        # detections = rescale_boxes(outputs, 416, (128, 128),)

        for count, detection in enumerate(detections):
            x1, y1, x2, y2 = get_baby_box(detection, args.img_size)

            with h5py.File(args.hdf5_path, 'a') as db:
                bboxes = db['bbox']
                for i in range(128):
                    bboxes[batch_i * 128 * args.batch_size + count * 128 + i, 0] = y1
                    bboxes[batch_i * 128 * args.batch_size + count * 128 + i, 1] = y2
                    bboxes[batch_i * 128 * args.batch_size + count * 128 + i, 2] = x1
                    bboxes[batch_i * 128 * args.batch_size + count * 128 + i, 3] = x2

            print('\nBaby bounding box added to file')
