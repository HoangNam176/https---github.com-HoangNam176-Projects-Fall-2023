import os
import argparse
import dlib
import cv2
import shutil
import math
import gc
import pandas as pd
import numpy as np
from copy import deepcopy
from imutils import face_utils


class Detection: # Detection model use Retina Face
    def __init__(self):
        caffemodel = "/home/anlab/Documents/Face_Anti_Spoofing/spoofing/resources/detection_model/Widerface-RetinaFace.caffemodel"
        deploy = "/home/anlab/Documents/Face_Anti_Spoofing/spoofing/resources/detection_model/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return bbox


path = "/home/anlab/Documents/Face_Anti_Spoofing/spoofing/resources/landmarks_models/shape_predictor_68_face_landmarks.dat"
detector = Detection()
predictor = dlib.shape_predictor(path) # Load model by dlib


def add_arguments():
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--folder_name",
        type=str,
        default="/home/anlab/Documents/Face_Anti_Spoofing/spoofing/landmarks/landmarks_image/folder_1688120772093",
        help="folder image used to test"
    )
    parser.add_argument(
        "--file_name_txt",
        type=str,
        default='/home/anlab/Documents/Face_Anti_Spoofing/spoofing/landmarks/landmarks_txt/folder_1688119582622/out_1688119583181.txt',
        help="file name to test landmarks"
    )
    return parser.parse_args()


def get_facial_landmarks(image):
    """_summary_
        Get facial landmarks with library dlib
    Args:
        image (_type_): image numpy

    Returns:
        _type_: Information facial landmarks
    """
    # Detection -> Predict facial landmarks
    image_bbox = detector.get_bbox(image)
    left, top, width, height = image_bbox
    right, bottom = left + width, top + height
    # Convert bounding box numpy to rectangle dlib
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bbox = dlib.rectangle(left, top, right, bottom)
    shape = predictor(image, bbox)
    shape = face_utils.shape_to_np(shape)
    return shape


def get_facial_landmarks_3d(image):
    import mediapipe as mp
    height, width = image.shape[:2]
    mp_face_mesh = mp.solutions.face_mesh # Load model by mediapipe face mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                    refine_landmarks=True,
                                    max_num_faces=1,
                                    min_detection_confidence=0.5)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for face_landmarks in results.multi_face_landmarks:
        keypoints = []
        for dataPoint in face_landmarks.landmark:
            keypoints.append([int(dataPoint.x * width), int(dataPoint.y * height)])
    
    del face_mesh, mp_face_mesh, mp
    gc.collect()
    return keypoints    


def save_folder_and_rotate(folder):
    name_folder = folder.split('/')[-1]
    files = os.listdir(folder)
    for file in files:
        img = cv2.imread(os.path.join(folder, file))
        if 'front' in name_folder:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        print(f'{folder}/{file}')
        cv2.imwrite(f'{folder}/{file}', img)


def copy_folder(src_folder, dst_folder):
    shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)


def remove_folder(folder):
    shutil.rmtree(folder)


def get_txt_landmarks(folder_name):
    name_folder = folder_name.split('/')[-1]
    if not os.path.exists(f'/home/anlab/Documents/Face_Anti_Spoofing/spoofing/landmarks/landmarks_txt/{name_folder}'):
        os.makedirs(f'/home/anlab/Documents/Face_Anti_Spoofing/spoofing/landmarks/landmarks_txt/{name_folder}')
    
    for file in os.listdir(folder_name):
        rows = []
        name, ext = file.split('.')
        path_image = os.path.join(folder_name, file)
        image = cv2.imread(path_image)
        landmarks = get_facial_landmarks_3d(image)
        coord_file = []
        for index, (x, y) in enumerate(landmarks):
            coord_file.append([y, x])
            rows.append(str(x) + " " + str(y))
        with open(f'/home/anlab/Documents/Face_Anti_Spoofing/spoofing/landmarks/landmarks_txt/{name_folder}/{name}.txt', 'w') as file:
            for row in rows:
                file.write(row)
                file.write('\n')

    
def calculator_scale_landmarks(file):
    # Load file txt
    f = open(file, 'r')
    keypoints = {}
    for index, line in enumerate(f):
        line = line.replace('\n', '')
        coordinates_x, coordinates_y = line.split(' ')
        coordinates_x, coordinates_y = int(coordinates_x), int(coordinates_y)
        keypoints[str(index+1)] = [coordinates_x, coordinates_y]
        
    # Get some keypoint to calculator
    width_face = keypoints['17'][0] - keypoints['1'][0]
    height_face = keypoints['9'][1] - (keypoints['25'][1] + keypoints['20'][1]) // 2
    
    # Get keypoint of eyes
    distance_left = (keypoints['42'][1] - keypoints['38'][1] + keypoints['41'][1] - keypoints['39'][1])
    distance_right = (keypoints['48'][1] - keypoints['44'][1] + keypoints['47'][1] - keypoints['45'][1])
    mean_eyes = (distance_left + distance_right) // 4
    return mean_eyes , height_face


def calculator_scale_landmarks_3d(file):
    # Load file txt
    f = open(file, 'r')
    keypoints = {}
    for index, line in enumerate(f):
        line = line.replace('\n', '')
        coordinates_x, coordinates_y = line.split(' ')
        coordinates_x, coordinates_y = int(coordinates_x), int(coordinates_y)
        keypoints[str(index+1)] = [coordinates_x, coordinates_y]
            
    # Get keypoint eyes right and left
    distance_left = (keypoints['145'][1] + keypoints['146'][1] + keypoints['154'][1] - keypoints['161'][1] - keypoints['160'][1] - keypoints['159'][1])
    distance_right = (keypoints['381'][1] + keypoints['375'][1] + keypoints['374'][1] - keypoints['386'][1] - keypoints['387'][1] - keypoints['388'][1])
    mean_eyes = (distance_right + distance_left) // 6
    height_face = keypoints['153'][1] - keypoints['11'][1]
    return mean_eyes, height_face


def sortDates(file):
    timestamp = file.split('.')[0].replace('out_', '')
    return timestamp


def save_ratio_landmarks(folder):
    save_ratio = os.path.abspath('./landmarks/ratio')
    if not os.path.exists(save_ratio):
        os.makedirs(save_ratio)
        
    for sub_folder in os.listdir(folder):
        rows = []
        name = sub_folder
        sub_folder = os.path.join(folder, sub_folder)
        files = os.listdir(sub_folder)
        files.sort(key=sortDates)
        list_distance_eyes = []
        list_height_face = []
        for file in files:
            file = os.path.join(sub_folder, file)
            distance_eyes, height_face = calculator_scale_landmarks_3d(file)
            list_distance_eyes.append(distance_eyes)
            list_height_face.append(height_face)
        
        # Consider frame sequence
        for index in range(len(list_distance_eyes)):
            row = {
                'frame': index,
                'distance_eyes': list_distance_eyes[index],
                'height_face': list_height_face[index],
                'ratio': list_distance_eyes[index] / list_height_face[index]
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(f'{save_ratio}/{name}.csv', header=True, sep=',')


# if __name__ == '__main__':
    
#     src = '/home/anlab/Documents/Face_Anti_Spoofing/test'
#     dst = '/home/anlab/Documents/Face_Anti_Spoofing/spoofing/landmarks/landmarks_image'
#     for sub_folder in os.listdir(src):
#         dst_folder = os.path.join(dst, sub_folder)
#         sub_folder = os.path.join(src, sub_folder)
#         save_folder_and_rotate(sub_folder)
#         copy_folder(sub_folder, dst_folder)
        
    
#     # Get facial landmarks
#     folder = "/home/anlab/Documents/Face_Anti_Spoofing/test"
#     for sub_folder in os.listdir(folder):
#         sub_folder = os.path.join(folder, sub_folder)
#         get_txt_landmarks(sub_folder)
#         remove_folder(sub_folder)
    
#     folder = '/home/anlab/Documents/Face_Anti_Spoofing/spoofing/landmarks/landmarks_txt/'
#     save_ratio_landmarks(folder)
    
#     folder = '/home/anlab/Documents/Face_Anti_Spoofing/spoofing/landmarks/ratio'
#     rows = []
#     for sub_folder in os.listdir(folder):
#         sub_folder = os.path.join(folder, sub_folder)
#         df = pd.read_csv(sub_folder)
#         row = {
#             'folder_name': sub_folder.split('/')[-1],
#             'Deviation_distance_eyes': df['distance_eyes'].std(), 
#             'Deviation_ratio': df['ratio'].std()
#         }
#         rows.append(row)
    
#     df = pd.DataFrame(rows)
#     df.to_csv('Deviation.csv', header=True, sep=',')