import argparse
import time
import streamlit as st
from PIL import Image
import os
import numpy as np

import torch

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements,  non_max_suppression,  \
    scale_coords, xyxy2xywh, set_logging
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device,  time_synchronized


@st.cache
def load_model(weights, device, imgsz, half):
    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    if half:
        model.half()  # to FP16

    return model


@st.cache
@torch.no_grad()
def detect(model,  # model.pt path(s)
           source='data/images',  # file/dir/URL/glob, 0 for webcam
           imgsz=416,  # inference size (pixels)
           conf_thres=0.25,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=1000,
           line_thickness=2,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           half=False,  # use FP16 half-precision inference
           ):
    stride = int(model.stride.max())  #
    names = model.module.names if hasattr(
        model, 'module') else model.names
    device = select_device(device)
    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()
        res = []
        # Process detections
        for _, det in enumerate(pred):  # detections per image
            _, s, im0, _ = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                      ) / gn).view(-1).tolist()  # normalized xywh
                    # label format
                    line = [cls.item(), torch.tensor(xyxy).view(
                        1, 4).view(-1).tolist(), conf.item()]
                    res.append(line)

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors(
                        c, True), line_thickness=line_thickness)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

    return im0, res


def get_coords(bbox, max_w, max_h, i):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    x = int((max_w-width)//2 + (i % 4)*max_w) + 5
    y = int((max_h-height)//2 + (i//4)*max_h) + 5

    return x, y


@ st.cache
def generate_crops(opt, detections):
    filename = opt.source
    image = Image.open(filename)

    max_w = -1
    max_h = -1
    bboxes = []
    for prediction in detections:
        bboxes.append(prediction[1])
        pred = prediction[1]
        w = pred[2] - pred[0]
        h = pred[3] - pred[1]

        if w > max_w:
            max_w = w
        if h > max_h:
            max_h = h
    print(len(detections)//4)
    canvas_width = int((max_w)*4+10)
    canvas_height = int(((len(detections)-1)//4+1)*max_h+10)
    dest = Image.new('RGB', (canvas_width, canvas_height),
                     color=(255, 255, 255))
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        im_crop = image.crop((x1, y1, x2, y2))

        x, y = get_coords(bbox, max_w, max_h, i)
        # x = int((i % 4)*max_w+10)
        # y = int((i//4)*max_h+3)
        dest.paste(im_crop, (x, y))

    return dest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size',
                        type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000,
                        help='maximum detections per image')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--line-thickness', default=2,
                        type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    opt = parser.parse_args()
    #check_requirements(exclude=('tensorboard', 'thop'))

    st.title("Spoor AS - Object detection demo")

    st.sidebar.write('#### Please upload an image .')
    uploaded_file = st.sidebar.file_uploader('',
                                             type=['png', 'jpg', 'jpeg'],
                                             accept_multiple_files=False)

    sample_filenames = os.listdir('data/images')

    st.sidebar.write(
        '[Find additional information about Spoor.](https://spoor.ai)')
    confidence_threshold = st.sidebar.slider(
        'Confidence threshold: What is the minimum acceptable confidence level for displaying a bounding box?', 0.0, 1.0, 0.25, 0.01)
    opt.conf_thres = confidence_threshold
    overlap_threshold = st.sidebar.slider(
        'Overlap threshold: What is the maximum amount of overlap permitted between visible bounding boxes?', 0.0, 1.0, 0.4, 0.01)
    opt.iou_thres = overlap_threshold
    logo = Image.open('./logos/Spoor-logo.png')
    st.sidebar.image(logo,
                     use_column_width=False)

    show_crops = st.sidebar.checkbox(label="Try this option out!",
                                     help="If you select this option maybe you will see the detected birds individually")

    imageLocation = st.empty()
    buttonLocations = st.empty()
    cropsLocations = st.empty()

    model = load_model(opt.weights, opt.device, opt.imgsz, opt.half)

    if uploaded_file is not None:
        with open(os.path.join("data/images/", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        opt.source = 'data/images/' + uploaded_file.name
        imageLocation.image(uploaded_file, use_column_width=False,
                            caption='Uploaded image waiting to be analyzed',)
        click = buttonLocations.button(label="Let's see those birds!",
                                       help='By clicking this button the trained YOLOv5 model will predict and display the objects in the image')
        if click:
            opt.half = True
            img, detections = detect(model=model,
                                     source=opt.source,
                                     imgsz=opt.imgsz,
                                     conf_thres=opt.conf_thres,
                                     iou_thres=opt.iou_thres)
            img_RGB = img[..., ::-1].copy()
            imageLocation.image(img_RGB, use_column_width=False,
                                caption='Generated predictions using YOLOv5')

            if show_crops:
                image_crops = generate_crops(opt, detections)
                cropsLocations.image(
                    image_crops, caption="Here they are: some tiny smalls birds near the wind-farm!")
