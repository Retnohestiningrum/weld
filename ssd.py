import colorsys
import os
import time
import warnings

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from nets.ssd import SSD300
from utils.anchors import get_anchors
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image)
from utils.utils_bbox import BBoxUtility

warnings.filterwarnings("ignore")


class SSD(object):
    def __init__(self, confidence=0.5, nms_iou=0.45):
        self.model_path = r'E:\ssd-pytorch-master-streamlit\logs\4.pth'
        self.classes_path = r'E:\ssd-pytorch-master-streamlit\model_data\class.txt'
        self.input_shape = [640, 640]
        self.backbone = "resnet50"
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.anchors_size = [30, 60, 111, 162, 213, 264, 315]
        self.letterbox_image = False
        self.cuda = torch.cuda.is_available()

        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors = torch.from_numpy(get_anchors(
            self.input_shape, self.anchors_size, self.backbone)).type(torch.FloatTensor)
        if self.cuda:
            self.anchors = self.anchors.cuda()
        self.num_classes = self.num_classes + 1

        hsv_tuples = [(x / self.num_classes, 1., 1.)
                      for x in range(self.num_classes)]
        self.colors = [tuple(int(c * 255)
                             for c in colorsys.hsv_to_rgb(*x)) for x in hsv_tuples]

        self.bbox_util = BBoxUtility(self.num_classes)
        self.generate()

    def generate(self):
        self.net = SSD300(self.num_classes, self.backbone)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(
            self.model_path, map_location=device))
        self.net = self.net.eval()
        if self.cuda:
            self.net = torch.nn.DataParallel(self.net).cuda()
        print(f'{self.model_path} model loaded.')

    def detect_image(self, image, crop=False, count=False, return_info=False):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(
            image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(
            np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            results = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape, self.letterbox_image,
                                                nms_iou=self.nms_iou, confidence=self.confidence)

            if len(results[0]) <= 0:
                if return_info:
                    return image, []
                else:
                    return image

            top_label = np.array(results[0][:, 4], dtype='int32')
            top_conf = results[0][:, 5]
            top_boxes = results[0][:, :4]

        font = ImageFont.truetype(
            r'E:\ssd-pytorch-master-streamlit\model_data\simhei.ttf', size=max(int(0.03 * np.shape(image)[1]), 12))
        thickness = max(
            (np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)

        detections = []

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = f'{predicted_class} {score:.2f}'
            draw = ImageDraw.Draw(image)
            text_size = draw.textbbox((0, 0), label, font)
            text_origin = np.array(
                [left, top - text_size[3] if top - text_size[3] >= 0 else top + 1])

            for t in range(thickness):
                draw.rectangle([left + t, top + t, right - t,
                               bottom - t], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + np.array(
                [text_size[2]-text_size[0], text_size[3]-text_size[1]]))], fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

            detections.append([left, top, right, bottom, int(c), float(score)])

        if return_info:
            return image, detections
        else:
            return image
