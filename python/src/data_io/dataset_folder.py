import os
import os.path as osp
import cv2
import torch
import numpy as np
from torchvision import datasets

IMG_FORMATS = ['jpg', 'jpeg', 'png']
VID_FORMATS = ['mp4', 'mov']

def opencv_loader(path):
    img = cv2.imread(path)
    return img


class DatasetFolderFT(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                ft_width=10, ft_height=10, loader=opencv_loader):
        super(DatasetFolderFT, self).__init__(root, transform, target_transform, loader)
        self.root = root
        self.ft_width = ft_width
        self.ft_height = ft_height

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        # generate the FT picture of the sample
        ft_sample = generate_FT(sample)
        if sample is None:
            print('image is None --> ', path)
        if ft_sample is None:
            print('FT image is None -->', path)
        assert sample is not None

        ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        ft_sample = torch.from_numpy(ft_sample).float()
        ft_sample = torch.unsqueeze(ft_sample, 0)
        print(sample.shape)
        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured: %s' % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        print(sample.size())
        return sample, ft_sample, target
    
    
class Dataset_CelebA_Spoof():
    def __init__(self, root, file_label, transform=None, target_transform=None,
                ft_width=10, ft_height=10, loader=opencv_loader):
        self.root = root
        self.file_label = file_label
        self.transform = transform
        self.target_transform = target_transform
        self.ft_width = ft_width
        self.ft_height = ft_height
        self.load_image = loader
        self.samples = self.extract_data_from_label()
    
    # file use to extract 
    def extract_data_from_label(self):
        assert osp.exists(self.root), f"{self.root} is not found"
        assert osp.isdir(self.root), f"{self.root} is not directory"
        
        # Read label (path/to/image + class)
        data = []
        f = open(self.file_label, 'r')
        for line in f.readlines():
            path, target = line.replace('\n', '').split()
            path = osp.join(self.root, path)
            name = path.split('/')[-1].split('.')[0]
            path_bbox = path.replace(path.split('/')[-1], name + '_BB.txt')
            try:
                if osp.exists(path):
                    f_bbox = open(path_bbox, 'r')
                    for line in f_bbox.readlines():
                        x, y, w, h, score = line.split()
                        data.append([path, int(target), int(x), int(y), int(w), int(h)])
                        break
            except Exception as err:
                pass
        return data
    
    
    def __getitem__(self, index):
        path, target, left, top, width, height = self.samples[index]
        sample = self.load_image(path)
        bbox_cvt = self.xywhceleb_xywh([left, top, width, height], sample.shape[1], sample.shape[0])
        sample_face = crop_image_xywh(sample, bbox_cvt, 1.2)
        sample_face = cv2.resize(sample_face, (80, 80))
        # generate the FT picture of the sample
        ft_sample = generate_FT(sample_face)
        if sample is None:
            print('image is None --> ', path)
        if ft_sample is None:
            print('FT image is None -->', path)
        assert sample is not None

        ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        ft_sample = torch.from_numpy(ft_sample).float()
        ft_sample = torch.unsqueeze(ft_sample, 0)

        if self.transform is not None:
            try:
                sample_face = self.transform(sample_face)
            except Exception as err:
                print('Error Occured: %s' % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample_face, ft_sample, target
    
    
    def __len__(self):
        return len(self.samples)


    @staticmethod
    def xywhceleb_xywh(bbox, src_w, src_h):
        x1 = int(bbox[0]*(src_w / 224))
        y1 = int(bbox[1]*(src_h / 224))
        w1 = int(bbox[2]*(src_w / 224))
        h1 = int(bbox[3]*(src_h / 224))
        return x1, y1, w1, h1


def generate_FT(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+1)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn+1) / (maxx - minn+1)
    return fimg


def _get_new_box(src_w, src_h, bbox, scale):
    x = bbox[0]
    y = bbox[1]
    box_w = bbox[2]
    box_h = bbox[3]

    scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))

    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w/2+x, box_h/2+y

    left_top_x = center_x-new_width/2
    left_top_y = center_y-new_height/2
    right_bottom_x = center_x+new_width/2
    right_bottom_y = center_y+new_height/2

    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0

    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0

    if right_bottom_x > src_w-1:
        left_top_x -= right_bottom_x-src_w+1
        right_bottom_x = src_w-1

    if right_bottom_y > src_h-1:
        left_top_y -= right_bottom_y-src_h+1
        right_bottom_y = src_h-1

    return int(left_top_x), int(left_top_y),\
            int(right_bottom_x), int(right_bottom_y)


def crop_image_xywh(image, bbox, scale, crop=True):
    if crop:
        src_h, src_w, _ = np.shape(image)
        left_top_x, left_top_y, \
            right_bottom_x, right_bottom_y = _get_new_box(src_w, src_h, bbox, scale)
        # print(left_top_x, left_top_y, right_bottom_x, right_bottom_y)
        img = image[left_top_y: right_bottom_y + 1,
                    left_top_x: right_bottom_x + 1]
        return img
    else:
        return image
        

# if __name__ == '__main__':
#     path = '/home/hoang/Documents/work/Data/Face_Anti_Spoofing/CelebA_Spoof'
#     label = '/home/hoang/Documents/work/Data/Face_Anti_Spoofing/CelebA_Spoof/metas/intra_test/train_label.txt'
#     celeb_spoof = Dataset_CelebA_Spoof(path, label)
#     celeb_spoof[6]