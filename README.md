# RotatedObjectDetection

Train the Rotated Object Detection model for our own datasets.

This repository contain two Oriented Object Detection algorithms : YOLOV5_OOB and YOLOV7_OOB, which are modified by YOLOV5 and YOLOV7.

# Paper Links  

* [YOLOV7](https://arxiv.org/abs/2207.02696) : YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors

* [DOTA v2](https://arxiv.org/pdf/2102.12219.pdf) : Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges

* [KLD Loss](https://arxiv.org/pdf/2106.01883.pdf) : Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence

# Packages Version Need
```
torch                    2.1.1
torchvision              0.16.1
triton                   2.1.0
matplotlib               3.8.2
mpmath                   1.3.0
networkx                 3.2.1
numpy                    1.26.2
nvidia-cublas-cu12       12.1.3.1
nvidia-cuda-cupti-cu12   12.1.105
nvidia-cuda-nvrtc-cu12   12.1.105
nvidia-cuda-runtime-cu12 12.1.105
nvidia-cudnn-cu12        8.9.2.26
nvidia-cufft-cu12        11.0.2.54
nvidia-curand-cu12       10.3.2.106
nvidia-cusolver-cu12     11.4.5.107
nvidia-cusparse-cu12     12.1.0.106
nvidia-nccl-cu12         2.18.1
nvidia-nvjitlink-cu12    12.3.101
nvidia-nvtx-cu12         12.1.105
```

# setup environment of YOLOV7_OOB
__You need to have installed CUDA under Linux system__
```
cd utils/nms_rotated/
python setup.py build_ext --inplace
```

# label package
* [roLabelimg](https://github.com/cgvict/roLabelImg)

the format of roLabelimg is (cx,cy,w,h,theta)

Using roxml_to_dota.py:

for YOLOV7_OOB, you need to transform the format to dota format (x0,y0,x1,y1,x2,y2,x3,y3) (not normalized) 

for YOLOV5_OOB, you need to transform the format to yolov5 format (x0,y0,x1,y1,x2,y2,x3,y3) (normalized)

# data augmentation
we provide mixup and mosaic data augmentation method for Oriented Object Detection in rotate_aug.py.

example:
```
GetAugImgAndLabel(annotation_lines,aug_num=20,aug_path='/home/kingargroo/seed/rice_aug',dest_file='/home/kingargroo/seed/rice_aug_label')
```

The results of mixup can be visualized as :<br>
![image](https://github.com/ZeonlungPun/OrientedObjectDetection/blob/main/demo/20.jpg)

The results of mosaic can be visualized as :<br>
![image](https://github.com/ZeonlungPun/OrientedObjectDetection/blob/main/demo/7.jpg)

# finnal result demo
The results of Oriented Object Detection :<br>
![image](https://github.com/ZeonlungPun/OrientedObjectDetection/blob/main/demo/r1.png)

![image](https://github.com/ZeonlungPun/OrientedObjectDetection/blob/main/demo/r2.png)


