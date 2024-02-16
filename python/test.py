import os
import cv2
import numpy as np
import argparse
import warnings
import onnxruntime
import matplotlib.pyplot as plt
import pandas as pd
from skimage.feature import (corner_harris, corner_subpix, corner_peaks,
                             plot_matches)
from skimage.transform import warp, AffineTransform
from skimage.measure import ransac
from landmarks import get_landmarks
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
# from src.utils import make_folder, save_numpy_as_binary

# create dataset
# from src.utils import direct_face_from_rose_youtu 

warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/sample/"


def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def add_argument():
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
        default="./resources/anti_spoof_models_onnx",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        default=None,
        help="image used to test")
    parser.add_argument(
        "--folder_name",
        type=str,
        default="/home/anlab/Documents/Face_Anti_Spoofing/test/folder_1688120746578",
        help="folder image used to test"
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help='Log file include (file name and error label)'
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default=None,
        help='Save folder image'
    )
    return parser.parse_args() 


def test_combine(image_name, model_dir, device_id, index):    
    # Check ext in image_name in ['jpg', 'jpeg', 'png']
    name, ext = image_name.split('/')[-1].split('.')
    if ext.lower() in ['jpg', 'jpeg', 'png']:
        # Load model
        model_test = AntiSpoofPredict(device_id)
        image_cropper = CropImage()
        image = cv2.imread(image_name)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = cv2.flip(image, 1)
        
        image_bbox = model_test.get_bbox(image)
              
        result_pred = []
        result_pred.append(image_bbox)
        # sum the prediction from single model's result
        prediction = np.zeros((1, 3))
        value = None
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": 1.2,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            model = onnxruntime.InferenceSession(os.path.join(model_dir, model_name))
            model_name = model_name.replace('.onnx', '')
            # save_numpy_as_binary(img, os.path.join('/content', model_name, name + '.bin'))
            inp_name = model.get_inputs()[0].name
            out_name = model.get_outputs()[0].name
            if 'MiniFASNet' in model_type:
                img = np.transpose(img, (2, 0, 1)).astype(np.float32)
                result = model.run([out_name], {inp_name: img})[0]
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.transpose(img, (2, 0, 1)).astype(np.float32)
                result = model.run([out_name], {inp_name: img})[0]
            if result.shape == (1, 3):
                value = result[0][0] + result[0][2]
            else:
                value = result[0][0]
                
            result_pred.append(value)
            
        return result_pred
    return ""


def test_model(image_name, model_name, device_id):
    if '80x80' in model_name:
        img = np.fromfile(image_name, dtype=np.uint8).reshape(80, 80, 3)
    else:
        img = np.fromfile(image_name, dtype=np.uint8).reshape(128, 128, 3)
    model = onnxruntime.InferenceSession(model_name)
    inp_name = model.get_inputs()[0].name
    out_name = model.get_outputs()[0].name
    value = None
    if  'MiniFASNet' in model_name:
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        result = model.run([out_name], {inp_name: img})[0]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        result = model.run([out_name], {inp_name: img})[0]
    
    if result.shape == (1, 3):
        value = result[0][0] + result[0][2]
    else:
        value = result[0][0]
    return value


def run_model_combine(args):
    rows = []
    folder, model_dir, device = args.folder_name, args.model_dir, args.device_id
    for index, f in enumerate(os.listdir(folder)):
        abs_path = os.path.join(folder, f)
        pred = test_combine(abs_path, model_dir, device, index)
        row = {
            'image_name': f,
            '1_128x128_MobileNetV3': round( 100 * pred[1], 2),
            '2.7_80x80_MiniFASNetV2': round( 100 * pred[2], 2),
            '1_80x80_MiniFASNetV2': round(100 * pred[3], 2),
            'score': round(100 * (pred[1] + pred[2] + pred[3]) / 3, 2)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv('score_combine_spoof.csv', header=True, sep=',')


def run_model_split(args):
    # List folder
    list_folder = [
        '/home/anlab/Documents/Face_Anti_Spoofing/1_80x80_MiniFASNetV2',
        '/home/anlab/Documents/Face_Anti_Spoofing/2.7_80x80_MiniFASNetV2',
        '/home/anlab/Documents/Face_Anti_Spoofing/1_128x128_MobileNetV3'
    ]

    list_model = [
        './resources/anti_spoof_models_onnx/1_80x80_MiniFASNetV2.onnx',
        './resources/anti_spoof_models_onnx/2.7_80x80_MiniFASNetV2.onnx',
        './resources/anti_spoof_models_onnx/1_128x128_MobileNetV3.onnx'
    ]

    # Get file name (append filename)
    filenames = []
    for f in os.listdir(list_folder[0]):
        filenames.append(f)

    # Inference
    rows = []
    for filename in filenames:
        pred = []
        for index, model_name in enumerate(list_model):
            base_filename = os.path.join(list_folder[index], filename)
            value = test_model(base_filename, model_name, args.device_id)
            pred.append(value)
        
        print(pred, filename)
        row = {
            'image_name': filename.replace('.bin', ''),
            '1_128x128_MobileNetV3': round( 100 * pred[2], 2),
            '2.7_80x80_MiniFASNetV2': round( 100 * pred[1], 2),
            '1_80x80_MiniFASNetV2': round(100 * pred[0], 2),
            'score': round(100 * (pred[0] + pred[1] + pred[2]) / 3, 2)
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv('score_split.csv', header=True, sep=',')


def crop_face(image_name, ratio, device_id, args):
    # Load model
    name, ext = image_name.split('/')[-1].split('.')
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(image_name)
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image = cv2.flip(image, 1)
    
    image_bbox = model_test.get_bbox(image)
    left, top, width, height = image_bbox[0], image_bbox[1], image_bbox[2], image_bbox[3]
    right, bottom = left + width, top + height
    new_width = int (height / ratio) 
    deviate = width - new_width + 1 if (width - new_width) % 2 == 1 else width - new_width
    new_left = left + deviate // 2
    new_right = right - deviate // 2
    # Crop face image
    # img_crop_face = image[top:bottom, new_left:new_right, :]
    # Save folder
    for model_name in os.listdir(args.model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        model_name = model_name.replace('.onnx', '')
        param = {
            "org_img": image,
            "bbox": [new_left, top, new_right - new_left, bottom - top],
            "scale": 1.2,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        # cv2.imshow("image", img)
        # k = cv2.waitKey(0)
        save_numpy_as_binary(img, os.path.join('/home/anlab/Documents/Face_Anti_Spoofing', model_name, name + '.bin'))


def findHormography_scikit(image_src, image_dst, frame_src, frame_dst, coor_src, coor_dst):
    # estimate affine transform model using all coordinates
    coor_src = np.array(coor_src)
    coor_dst = np.array(coor_dst)
    model = AffineTransform()
    model.estimate(coor_src, coor_dst)

    # robustly estimate affine transform model with RANSAC
    model_robust, inliers = ransac((coor_src, coor_dst), AffineTransform, min_samples=3,
                                residual_threshold=2, max_trials=100)
    
    outliers = (inliers == False)    
    # visualize correspondence
    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.suptitle(f"Faulty and Correct correspondences with frame {frame_src} and frame {frame_dst}", fontsize=14)
    # Convert bgr to rgb (show image in plt)
    image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    image_dst = cv2.cvtColor(image_dst, cv2.COLOR_BGR2RGB)
    
    inlier_idxs = np.nonzero(inliers)[0]
    plot_matches(ax[0], image_src, image_dst, coor_src, coor_dst,
                np.column_stack((inlier_idxs, inlier_idxs)), matches_color='b')
    ax[0].axis('off')
    ax[0].set_title(f'Correct correspondences : {list(inliers).count(True)}')

    outlier_idxs = np.nonzero(outliers)[0]
    plot_matches(ax[1], image_src, image_dst, coor_src, coor_dst,
                np.column_stack((outlier_idxs, outlier_idxs)), matches_color='r')
    ax[1].axis('off')
    ax[1].set_title(f'Faulty correspondences: {list(outliers).count(True)}')
    # plt.show()
    plt.savefig(f'{frame_src}.png')
    return list(outliers).count(True), list(inliers).count(True)


def findHormography_opencv(image_src, image_dst, frame_src, frame_dst, coor_src, coor_dst):
    # Calculate homography
    coor_src = np.array(coor_src)
    coor_dst = np.array(coor_dst)
    h, inliers = cv2.findHomography(coor_src, coor_dst, cv2.RANSAC,8.0)
    inliers = inliers.reshape(1, -1)[0]
    
    # Convert 1: True, 0: False
    inliers[inliers == 0] = False
    inliers[inliers == 1] = True
    
    outliers = (inliers == False)
    # visualize correspondence
    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.suptitle(f"Faulty and Correct correspondences with frame {frame_src} and frame {frame_dst}", fontsize=14)
    # Convert bgr to rgb (show image in plt)
    image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    image_dst = cv2.cvtColor(image_dst, cv2.COLOR_BGR2RGB)
    
    inlier_idxs = np.nonzero(inliers)[0]
    plot_matches(ax[0], image_src, image_dst, coor_src, coor_dst,
                np.column_stack((inlier_idxs, inlier_idxs)), matches_color='b')
    ax[0].axis('off')
    ax[0].set_title(f'Correct correspondences : {list(inliers).count(True)}')

    outlier_idxs = np.nonzero(outliers)[0]
    plot_matches(ax[1], image_src, image_dst, coor_src, coor_dst,
                np.column_stack((outlier_idxs, outlier_idxs)), matches_color='r')
    ax[1].axis('off')
    ax[1].set_title(f'Faulty correspondences: {list(outliers).count(True)}')
    plt.show()
    plt.savefig(f'{frame_src}.png')
    return list(outliers).count(True), list(inliers).count(True)


if __name__ == '__main__':
    
    path_image_1 = '/home/anlab/Documents/Spoof/single_frame/figure_far/pair9/131.jpg'
    path_image_2 = '/home/anlab/Documents/Spoof/single_frame/figure_far/pair9/IMG_20230713_090633.jpg'
    frame_src, frame_dst = path_image_1.split('/')[-1], path_image_2.split('/')[-1]
    image_1 = cv2.imread(path_image_1)
    image_1 = cv2.resize(image_1, (480, 640))
    image_2 = cv2.imread(path_image_2)
    image_2 = cv2.resize(image_2, (480, 640))
    
    # cv2.imshow('frame', image_2)
    # k = cv2.waitKey(0)
    
    coord_image_1 = get_landmarks(image_1)
    coord_image_2 = get_landmarks(image_2)
    
    findHormography_opencv(image_1, image_2, frame_src, frame_dst, coord_image_1, coord_image_2)