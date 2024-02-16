"""_summary_
    Test model python
    
"""
# use code ddddeer
import os
import cv2
import numpy as np
import argparse
import warnings
import time
import pandas as pd

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/sample/"


# å› ä¸ºå®‰å�“ç«¯APKè�·å�–ç�„è§†é¢‘æµ�å®½é«˜æ¯”ä¸º3:4,ä¸ºäº†ä¸�ä¹‹ä¸€è‡´ï¼Œæ‰€ä»¥å°†å®½é«˜æ¯”é™�åˆ¶ä¸º3:4
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def test(image_name, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(image_name)
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image = cv2.flip(image, 1)
    image_bbox = model_test.get_bbox(image)
    test_speed = 0
    # sum the prediction from single model's result
    result_pred = []
    prediction = np.zeros((1, 3))
    value = None
    for model_name in os.listdir(model_dir):
        print(model_name)
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": 1.2,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        if scale == 1:
            result = model_test.predict(img, os.path.join(model_dir, model_name), num_class=2, bias_prob=True)
            result = np.append(result, [[0]])
            prediction += result
        else:
            result = model_test.predict(img, os.path.join(model_dir, model_name), num_class=3, bias_prob=False)
            prediction += result
        if result.shape == (1, 3):
            value = result[0][0] + result[0][2]
        else:
            value = result[0]
        
        result_pred.append(value)
    return result_pred

if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        default="/home/anlab/Face_Anti_Spoofing/spoofing/datasets/Data/train_v1/2.7_80x80/1/5s_2402.jpg",
        help="image used to test")
    parser.add_argument(
        "--folder_name",
        type=str,
        default="/home/anlab/Documents/Face_Anti_Spoofing/test/test_new",
        help="folder image used to test"
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help='Log file include (file name and error label)'
    )
    
    
    
    args = parser.parse_args()
    label_pred = 0
    rows = []
    files = os.listdir(args.folder_name)
    for index, file_name in enumerate(os.listdir(args.folder_name)):
        path = os.path.join(args.folder_name, file_name)
        pred = test(path, args.model_dir, args.device_id)
        print(pred)
        row = {
            "image_name": file_name,
            "1_80x80_MiniFASNetV2": round(pred[0] * 100, 2),
            "2.7_80x80_MiniFASNetV2": round(pred[1] * 100, 2),
            'score': round(100 * (pred[0] + pred[1]) / 2, 2)
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.mean(axis=0))
    df.to_csv("score_python.csv", header=True, index=False, sep='\t') 