import os.path as osp
import os
import struct
import glob
import cv2
import dlib
import math
import numpy as np
import sys
import logging
from imutils import face_utils
from attrdict import AttrDict as adict
from importlib import import_module
from src.generate_patches import CropImage

IMG_FORMATS = ['jpg', 'jpeg', 'png', 'giff']

ROTATE_DEGREE = {
    "0": None,
    "90": cv2.ROTATE_90_CLOCKWISE,
    "-90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180
}

# Load model landmarks and detection
# p = our pre-treined model directory, on my case, it's on the same script's diretory.


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
predictor = dlib.shape_predictor(path)


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))
    
def check_folder_exist(foldername, msg_tmpl='folder "{}" does not exist'):
    if not osp.isdir(foldername):
        raise FileNotFoundError(msg_tmpl.format(foldername))


def make_folder(foldername):
    if not osp.exists(foldername):
        os.makedirs(foldername)
    else:
        # Warning foldername is error
        logging.warning(f'{foldername} is exsits.')


def read_py_config(filename):
    filename = osp.abspath(osp.expanduser(filename))
    check_file_exist(filename)
    assert filename.endswith('.py')
    module_name = osp.basename(filename)[:-3]
    if '.' in module_name:
        raise ValueError('Dots are not allowed in config file path.')
    config_dir = osp.dirname(filename)
    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = adict({
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
    })

    return cfg_dict


def PSNR(image1, image2):
    
    mse = np.mean((image1 - image2) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def save_image_as_binary(image_path, output_path):
    # Đọc nội dung của file ảnh
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    # Ghi dữ liệu vào file nhị phân
    with open(output_path, 'wb') as binary_file:
        # Ghi kích thước của dữ liệu
        binary_file.write(struct.pack('I', len(image_data)))
        # Ghi dữ liệu nhị phân
        binary_file.write(image_data)


def save_numpy_as_binary(image, output_path):
    # Chuyển đổi mảng numpy thành dạng nhị phân
    binary_data = struct.pack('B' * image.size, *image.flatten())


    # Lưu dữ liệu nhị phân vào file
    with open(output_path, 'wb') as binary_file:
        binary_file.write(binary_data)


def read_image_from_binary(binary_path):
    # Đọc dữ liệu từ file nhị phân
    with open(binary_path, 'rb') as binary_file:
        # Đọc kích thước của dữ liệu
        size = struct.unpack('I', binary_file.read(4))[0]
        # Đọc dữ liệu nhị phân
        image_data = binary_file.read(size)

    # Chuyển đổi dữ liệu thành mảng numpy
    image_array = np.frombuffer(image_data, dtype=np.uint8)

    # Chuyển đổi mảng numpy thành ảnh OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return image


def create_dataset_NUAA(source, target):
    
    detect_face = Detection()
    image_cropper = CropImage()
    
    sourcename = osp.abspath(osp.expanduser(source))
    targetname = osp.abspath(osp.expanduser(target))
    print(targetname)
    check_folder_exist(sourcename)
    
    # Create dir
    if not osp.exists(targetname):
        os.makedirs(os.path.join(targetname, 'real'))
        os.makedirs(os.path.join(targetname, 'spoof'))
        
    real = osp.join(source, 'ClientRaw')
    spoof = osp.join(source, 'ImposterRaw')
    
    # Create dataset face for clientRaw
    for dirpath, dirs, files in os.walk(real):
        for file in files:
            name, ext = file.split('.')
            if ext in IMG_FORMATS:
                file = osp.join(dirpath, file)
                img = cv2.imread(file)
                bbox = detect_face.get_bbox(img)
                param = {
                    "org_img": img,
                    "bbox": bbox,
                    "scale": 1.2,
                    "out_w": 80,
                    "out_h": 80,
                    "crop": True,
                }
                img_crop = image_cropper.crop(**param)
                file_out = os.path.join(targetname, 'real', name + '.jpg')
                cv2.imwrite(file_out, img_crop)
                
    
    # Create dataset face for clientRaw
    for dirpath, dirs, files in os.walk(spoof):
        for file in files:
            name, ext = file.split('.')
            if ext in IMG_FORMATS:
                file = osp.join(dirpath, file)
                img = cv2.imread(file)
                bbox = detect_face.get_bbox(img)
                param = {
                    "org_img": img,
                    "bbox": bbox,
                    "scale": 1.2,
                    "out_w": 80,
                    "out_h": 80,
                    "crop": True,
                }
                img_crop = image_cropper.crop(**param)
                file_out = os.path.join(targetname, 'spoof', name + '.jpg')
                cv2.imwrite(file_out, img_crop)
     
                
def save_image_with_rotate(source, dist, rotate, flip_horizotal):

    # Check conditional source
    assert os.path.exists(source), f"{source} not found"
    assert os.path.isdir(source), f"{source} is not dir"
    
    make_folder(dist)
    
    for file in os.listdir(source):
        path_img = os.path.join(source, file)
        img = cv2.imread(path_img)
        if rotate != '0':
            img = cv2.rotate(img, ROTATE_DEGREE[rotate])
        if flip_horizotal:
            img = cv2.flip(img, 1)
            
        # Save file image
        path_img_save = os.path.join(dist, file)
        cv2.imwrite(path_img_save, img)


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


def direct_face_video(path_video):
    vid_capture = cv2.VideoCapture(path_video)
    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
    
    # while (vid_capture.isOpend()):
    #     ret, frame = vid_capture.read()
    #     if ret:
    #         for ro
    #     else:
    #         break

def direct_face_from_rose_youtu(folder_rose_youtu):
    """_summary_
        Direct face
    Args:
        folder_rose_youtu (_type_): folder include dataset ROSE-YOUTU
    """
    assert os.path.exists(folder_rose_youtu), f"{folder_rose_youtu} is not found"
    assert os.path.isdir(folder_rose_youtu), f"{folder_rose_youtu} is not directory"
    
    degrees = {
        'vid_name': [],
        'vid_degree': []
    }
    
    for path_vid in glob.glob(folder_rose_youtu + '/**/*.mp4'):
        direct_face_video(path_vid)