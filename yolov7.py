import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel



def show_in_moved_window(win_name, img, x, y, cmap=None):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    img = cv2.resize(img, (660, 440)) 
    if cmap == None:
        cv2.imshow(win_name, img)
    else:
        cv2.imshow(win_name, img, cmap=cmap)



def createModel():
    print('in here')
    weights = ['yolov7.pt']

    imgsz = 640
    trace = True

    device = select_device('')

    # Load model
    with torch.no_grad():
        model = attempt_load(weights, map_location=device)  # load FP32 model
    
    if trace:
        model = TracedModel(model, device, imgsz)

    print('### yolov7 model loaded!')
    
    return model

def predictwyolo(model, source):
    with torch.no_grad():
        return predict(model, source)


def predict(model, source):

    device = select_device('')
    imgsz = 1280


    conf_thres = 0.50
    iou_thres = 0.45

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size



    # -----create dataset------
    im0s = source
    # im0s = cv2.imread(source)  # BGR
    # Padded resize
    img = letterbox(im0s, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    # -----create dataset------



    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # print(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


    t0 = time.time()
    # for path, img, im0s, vid_cap in dataset2:
    # print('in loop')
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    augment = False
    # Inference
    t1 = time_synchronized()
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=augment)[0]
    t2 = time_synchronized()


    # classes = None # use all the classes
    classes = [0, 1, 2, 3, 4] # only use these classes
    agnostic_nms = False
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t3 = time_synchronized()

    identified = []
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s, im0 = '', im0s

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # label = f'{names[int(cls)]} {conf:.2f}'
                # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # print(xyxy)
                # print(names[int(cls)])
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                # print(c1, c2)


                identified.append({"x": (c1[0] + c2[0])/2,
                            "y": (c1[1] + c2[1])/2,
                            "width": abs(c1[0] - c2[0]),
                            "height": abs(c1[1] - c2[1]),
                            "class": f'{names[int(cls)]}',
                            "confidence": f'{conf:.2f}',
                            })

        # Print time (inference + NMS)
        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')


    # print(f'Done. ({time.time() - t0:.3f}s)')
    return identified


def drawClassifiedObjects(image, predictions):
    new_image = image.copy()

    color = (255, 0, 0)
    stroke = 2

    for prediction in predictions:
        if prediction["class"] == "wheel loader":
            continue

        # Get different dimensions/coordinates
        x = prediction["x"]
        y = prediction["y"]
        width = prediction["width"]
        height = prediction["height"]
        class_name = prediction["class"].lower()
        # Draw bounding boxes for object detection prediction
        cv2.rectangle(new_image, (int(x - width / 2), int(y + height / 2)), (int(x + width / 2), int(y - height / 2)), color, 2)
        # Get size of text
        text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]

        # Draw center of object
        new_image = cv2.circle(new_image, (int(x),int(y)), radius=2, color=(0, 0, 255), thickness=-1)

        # Write text onto image
        cv2.putText(new_image, class_name, (int(x - width / 2), int(y - height / 2 + text_size[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)

    return new_image


def ObtainCenter(image, predictions):

    for prediction in predictions:
        if prediction["class"] == "wheel loader":
            continue

        # Get different dimensions/coordinates
        x = prediction["x"]
        y = prediction["y"]

        
    return x, y


if __name__ == '__main__':

    source = 'test.jpg'
    source = cv2.imread(source)  # BGR

    model = createModel()
    print('after setup')
    pred = predictwyolo(model, source)
    print(pred)