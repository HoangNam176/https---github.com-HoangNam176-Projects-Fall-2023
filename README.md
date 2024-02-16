# F[a](https://github.com/anlab-ai/face-anti-spoofing)ce Anti Spoofing

## Installation

- Download [Anaconda](https://www.anaconda.com/products/individual) if it is not installed on your machine
- Clone the repository

```python
git clone https://github.com/HoangNam176/Projects-Fall-2023.git
```

- Create a project environment

```python
cd gymdx-multi-cam-tracking
conda create --name py37 python=3.7
conda activate py37
```

- Install dependencies

```python
pip install -r requirements.txt
```

## Face anti spoof step

### Step 1: Reasearch code (train and test model) use code python

#### Prepare Data

- To classify real or fake images based on the MobileNetv1 and MobileNetV2 models, it is necessary to have image data containing real images and fake images. Here, I use the CelebA-Spoof data set to conduct training (about 50GB).
- Link: [https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z](https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z)
- Organize the data file used for training and testing as follows:

```text
|data
|_____|train
|__________|object_1
|__________________|real
|____________________|image1.jpg
|____________________|image2.jpg
|__________________|fake
|____________________|image1.jpg
|____________________|image2.jpg
|__________|object_2
|__________________|real
|____________________|image1.jpg
|____________________|image2.jpg
|__________________|fake
|____________________|image1.jpg
|____________________|image2.jpg
|_____|test
|__________|object_1
|__________________|real
|____________________|image1.jpg
|____________________|image2.jpg
|__________________|fake
|____________________|image1.jpg
|____________________|image2.jpg
|__________|object_2
|__________________|real
|____________________|image1.jpg
|____________________|image2.jpg
|__________________|fake
|____________________|image1.jpg
|____________________|image2.jpg

Each image is resized to 80x80 size
```

#### Train model + Test model

- Go to the ``default_config.py`` file to change some parameters for training and testing such as training path, learning rate, etc.
- Command to train model real/fake classification:

```python
cd python/
python train.py --device_ids 0 --use_pretraind --patch_info 1_80x80
```

### Convert model train .pth -> ncnn (inference for app mobile)

#### Convert model .pth -> .onnx

- To convert a model from .pth format to .onnx format, go to the ``convert_model.py`` file to edit the input and output paths.
- After Change input path and output path, run command:

```python
python convert_model.py
```

#### Convert model .onnx -> .ncnn

- If have file .onnx after convert, access link ``https://convertmodel.com/`` to convert ncnn. After converted ncnn and downloads local computer

### Demo in App Mobile

#### Install Android Studio for Ubuntu

- Use link : [https://funix.edu.vn/chia-se-kien-thuc/cach-cai-dat-android-studio-tren-ubuntu/](https://funix.edu.vn/chia-se-kien-thuc/cach-cai-dat-android-studio-tren-ubuntu/) to install Android Studio
- Get model .ncnn in step **convert model** to app/Silent-Face-Anti-Spoofing-APK/engine/src/main/assets/live/
- Demo

  - Image real: ![Image real](https://i.imgur.com/NekVSCI.jpeg)
  - Image fake use Computer: ![Image fake computer](https://i.imgur.com/74sTuDk.jpeg)
  - Image fake use Mobile: ![Image fake Mobile](https://i.imgur.com/KxeQfmA.jpeg)
  - Image fake use A4 poster: ![Image fake A4 Poster](https://i.imgur.com/yS0eFl6.jpeg)

```python
python demo.py --videos videos\init\Double1.mp4 videos\init\Single1.mp4 --version v3
```
