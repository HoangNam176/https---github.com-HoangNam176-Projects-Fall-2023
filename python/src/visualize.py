import numpy as np
import cv2

def draw_bbox(img, bbox):
    """
        Draw bounding box: support for bbox
    """
    src_h, src_w, _ = img.shape

    xmin, ymin, box_w, box_h = bbox

    xmin = xmin if xmin > 0 else 0
    ymin = ymin if ymin > 0 else 0
    xmax = src_w if (xmin + box_w) > src_w else (xmin + box_w)
    ymax = src_h if (ymin + box_h) > src_h else (ymin + box_h)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return img
    