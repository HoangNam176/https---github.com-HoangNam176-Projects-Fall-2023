from datetime import datetime
import os
import math
import cv2
import time
import numpy as np
import pandas as pd


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


def get_width_height(patch_info):
    w_input = int(patch_info.split('x')[-1]) # w image
    h_input = int(patch_info.split('x')[0].split('_')[-1]) # h image
    return w_input,h_input


def parse_model_name(model_name):
    info = model_name.split('_')[0:-1]
    print(info)
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]
    print(model_type)
    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    
    return int(h_input), int(w_input), model_type, scale


def trignometry_for_distance(a, b):
    return math.sqrt(((b[0] - a[0]) * (b[0] - a[0])) +\
                     ((b[1] - a[1]) * (b[1] - a[1])))

# Find eyes
def rotate_by_angle(eyes):
 
    # Current get bounding box eyes have max conf class
    
    # Init left and right eye center
    right_eye_center = eyes[1]
    left_eye_center = eyes[0]
    
    # Coordinates right eye
    right_eye_x = right_eye_center[0]
    right_eye_y = right_eye_center[1]

    # Coordinates left eye
    left_eye_x = left_eye_center[0]
    left_eye_y = left_eye_center[1]

    # finding rotation direction
    if left_eye_y > right_eye_y:
        # print("Rotate image to clock direction")
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate image direction to clock
    else:
        # print("Rotate to inverse clock direction")
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    a = trignometry_for_distance(left_eye_center,
                                    point_3rd)
    b = trignometry_for_distance(right_eye_center,
                                    point_3rd)
    c = trignometry_for_distance(right_eye_center,
                                    left_eye_center)
    cos_a = (b*b + c*c - a*a)/(2*b*c)
    angle = (np.arccos(cos_a) * 180) / math.pi

    if direction == -1:
        angle = 90 - angle
    else:
        angle = -(90-angle)
    return angle


def rotate_by_coordinates(landmarks):

    left_eye, right_eye, nose, left_mouse, right_mouse = landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4]

    # Left eye
    left_eye_x, left_eye_y = left_eye[0], left_eye[1]
    right_eye_x, right_eye_y = right_eye[0], right_eye[1] 
    left_mouse_x, left_mouse_y = left_mouse[0], left_mouse[1] 
    right_mouse_x, right_mouse_y = right_mouse[0], right_mouse[1] 

    # Forward
    if (left_mouse_y > left_eye_y and left_mouse_y > right_eye_y) \
        and (right_mouse_y > left_eye_y and right_mouse_y > right_eye_y):
        return 0

    # Backward
    if (left_mouse_y < left_eye_y and left_mouse_y < right_eye_y) \
        and (right_mouse_y < left_eye_y and right_mouse_y < right_eye_y):
        return 180

    # Rotate -90 degree
    if (left_mouse_x > left_eye_x and left_mouse_x > right_eye_x) \
        and (right_mouse_x > left_eye_x and right_mouse_x > right_eye_x):
        return -90

    # Rotate 90 degree
    if (left_mouse_x < left_eye_x and left_mouse_x < right_eye_x) \
        and (right_mouse_x < left_eye_x and right_mouse_x < right_eye_x):
        return 90

    return 0 


# Preprocessing image with MobileNetV3
def preprocessing_img(img):
    mean = np.array([0.5931, 0.4690, 0.4229]).reshape((3, 1, 1))
    std = np.array([0.2471, 0.2214, 0.2157]).reshape((3, 1, 1))

    height, width = 128, 128
    img = cv2.resize(img, (height, width) , interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = img/255
    img = (img - mean)/std
    img = img.reshape((1, 3, height, width))
    return img.astype('float32')


def detect_blur_fft(image, size=60, threshold=10):
    """_summary_
        Detect blur in image 
    Args:
        image (_type_): _description_
        size (int, optional): _description_. Defaults to 60.
        threshold (int, optional): _description_. Defaults to 10.
        vis (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    h, w = image.shape
    (cx, cy) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    fftShift[cy - size : cy + size, cx - size : cx + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return (mean, mean <= threshold)


def deblur_image(image):
    # Chuyển ảnh sang ảnh grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tạo kernel Gaussian để áp dụng deconvolution
    kernel = cv2.getGaussianKernel(4, 0)
    kernel = kernel * kernel.T

    # Áp dụng phép biến đổi Fourier cho ảnh grayscale và kernel
    gray_fft = np.fft.fft2(gray)
    kernel_fft = np.fft.fft2(kernel)

    # Thực hiện deconvolution bằng phép chia trong miền tần số
    deblurred_fft = gray_fft / kernel_fft

    # Trở lại không gian ảnh
    deblurred = np.fft.ifft2(deblurred_fft)

    # Chuẩn hóa giá trị để đảm bảo nằm trong phạm vi 0-255
    deblurred = np.real(deblurred)
    deblurred = np.clip(deblurred, 0, 255).astype(np.uint8)
    return deblurred


def make_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

if __name__ == '__main__':
    
    folder = '/home/anlab/Downloads/test_anti_spoofing_1/test_bbox_spoof'
    for file in os.listdir(folder):
        name, ext = file.split('.')
        file = os.path.join(folder, file)
        img = cv2.imread(file)
        img = cv2.resize(img, (1280, 960))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        _, blur = detect_blur_fft(gray)
        print(name, _, blur)                