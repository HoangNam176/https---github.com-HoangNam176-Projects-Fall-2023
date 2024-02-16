import os
import cv2
import numpy as np
import argparse
import warnings
import time
import glob

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name, Face_Alignment

# Load model
model = AntiSpoofPredict(device_id=0) # GPU
image_cropper = CropImage() # Crop image


# Create folder image 80x80 only face from celeb-A
def create_data_from_celebA_spoof(source: str, dist: str, scale:str):
    # Create data base CelebA-Spoof
    try:
        if not os.path.exists(source):
            raise Exception
    
        # dist not found => make dirs
        if not os.path.exists(dist):
            os.makedirs(os.path.join(dist, f'{scale}_80x80', 'spoof'))
            os.makedirs(os.path.join(dist, f'{scale}_80x80', 'live'))
            
                    
        folder = os.path.join(dist, f'{scale}_80x80')
        files_id = os.listdir(source)
        for id in files_id:
            abs_id = os.path.join(source, id) # id include live and spoof
            classes = os.listdir(abs_id)
            for cls in classes:
                path_images = os.path.join(abs_id, cls)
                for image in os.listdir(path_images):
                    path_image = os.path.join(path_images, image)
                    name, ext = path_image.split('.')
                    try:
                        # Read image (cv2)
                        if ext in ['png', 'jpeg', 'jpg']:
                            img = cv2.imread(path_image)
                            image_bbox = model.get_bbox(img)
                            if image_bbox[2] > 10 and image_bbox[3] > 10:
                                param = {
                                    "org_img": img,
                                    "bbox": image_bbox,
                                    "scale": float(1),
                                    "out_w": int(80),
                                    "out_h": int(80),
                                    "crop": True,
                                }
                                img_save = image_cropper.crop(**param)
                                save_img = os.path.join(folder, cls, image)
                                cv2.imwrite(save_img, img_save) # Save img
                    except Exception as err:
                        print(err)
    except Exception as err:
        print(err)
        

# Convert data to Silent Face Anti Spoofing
def convert_data_from_Rose_Youtu(source: str, dist: str, scale: str, height: str, width: str):
    """_summary_
    Args:
        source (str): Data format by video
        dist (str): Save Data
        scale (str): scale image
    """
    try:
        if not os.path.exists(source):
            raise Exception

        if not os.path.exists(dist):
            os.makedirs(os.path.join(dist, f'{scale}_{height}x{width}', '0'))
            os.makedirs(os.path.join(dist, f'{scale}_{height}x{width}', '1'))

        
        dist_folder = os.path.join(dist, f'{scale}_{height}x{width}')
        list_id_video = os.listdir(source)
        index_image = 1
        for id_video in list_id_video:
            files = glob.glob(os.path.join(source, id_video, '*.mp4')) # Creating a glob object
            for file in files:
                name = file.split('/')[-1]
                typ, _, device, *other = name.split('_')
                video = cv2.VideoCapture(file)
                while video.isOpened():
                    ret, frame = video.read()
                    save_file = f'{dist_folder}/1/{device}_{index_image}.jpg' if typ == 'G' \
                                else f'{dist_folder}/0/{device}_{index_image}.jpg'
                    if ret:
                        if scale == 'org':
                            cv2.imwrite(save_file, frame)
                            index_image += 1
                        else:
                            image_bbox = model.get_bbox(frame)
                            if image_bbox[2] > 10 and image_bbox[3] > 10:
                                param = {
                                    "org_img": frame,
                                    "bbox": image_bbox,
                                    "scale": float(scale),
                                    "out_w": int(width),
                                    "out_h": int(height),
                                    "crop": True,
                                }
                                img_save = image_cropper.crop(**param)
                                cv2.imwrite(save_file, img_save)
                                index_image += 1
                    else:
                        break
        
    except Exception as err:
        print(err)


def orientation_object_in_video(file_name):
    vid = cv2.VideoCapture(file_name)
    
    # Load model eye haarcasde
    eye_detector = cv2.CascadeClassifier('/home/anlab/Face_Anti_Spoofing/spoofing/resources/detection_model/haarcascade_eye.xml')
    
    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            angle = Face_Alignment(eye_detector, frame)
            print(angle)
        else:
            break

if __name__ == '__main__':
    source = '/home/anlab/Downloads/ROSE-YOUTU/2/G_NT_5s_g_E_2_1.mp4'
    # dst = '/home/anlab/Face_Anti_Spoofing/spoofing/datasets/Data/org_Rose_Youtu'
    # convert_data_from_Rose_Youtu(source, dst, scale='org', height='', width='')
    orientation_object_in_video(source)
    