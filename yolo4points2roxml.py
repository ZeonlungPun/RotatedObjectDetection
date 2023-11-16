import os,json,cv2,math
import numpy as np
import pandas as pd
from collections import defaultdict

def coordinate_present_convert(coords, shift=True):
    """
    :param coords: shape [-1, 5]
    :param shift: [-90, 90) --> [-180, 0)
    :return: shape [-1, 5]
    """
    # angle range from [-90, 0) to [0,180)
    w, h = coords[:, 2], coords[:, 3]

    remain_mask = np.greater(w, h)
    convert_mask = np.logical_not(remain_mask).astype(np.int32)
    remain_mask = remain_mask.astype(np.int32)

    remain_coords = coords * np.reshape(remain_mask, [-1, 1])

    coords[:, [2, 3]] = coords[:, [3, 2]]
    coords[:, 4] += 90

    convert_coords = coords * np.reshape(convert_mask, [-1, 1])

    coords_new = remain_coords + convert_coords


    if shift:
        if coords_new[:, 4] >= 0:
            coords_new[:, 4] = 180 + coords_new[:, 4]

    return np.array(coords_new, dtype=np.float32)

def backward_convert(coordinate):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: format [x_c, y_c, w, h, theta, (label)]
    """
    boxes = []
    box = np.int0(coordinate)
    box = box.reshape([4, 2])
    rect1 = cv2.minAreaRect(box)

    x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]

    if theta == 0:
        w, h = h, w
        theta -= 90

    boxes.append([x, y, w, h, theta])

    return np.array(boxes, dtype=np.float32)

labels_list=['rice']
os.makedirs('/home/kingargroo/seed/roAnnotations', exist_ok=True)
print('建立roAnnotations目录')

raw_path_list=os.listdir('/home/kingargroo/seed/rpredict')
img_parent_path='/home/kingargroo/seed/residual'
for raw_path in raw_path_list:
    txt_path=os.path.join('/home/kingargroo/seed/rpredict',raw_path)
    main_path=txt_path.split('/')[-1].split('.')[0]
    img_path=os.path.join(img_parent_path,main_path)+'.jpg'
    img_name=main_path

    mem = defaultdict(list)

    fp=pd.read_csv(txt_path,header=None)
    for index,content in fp.iterrows():
        line=content.iloc[0]

        line = list(map(float, line.split(" ")))

        class_id=int(line[0])
        x=[float(line[1]),float(line[3]),float(line[5]),float(line[7])  ]
        y=[float(line[2]),float(line[4]),float(line[6]),float(line[8])  ]

        img = cv2.imread(img_path)
        height, width = img.shape[:-1]

        label = labels_list[class_id]
        x1 = min(x)
        x2 = max(x)
        y1 = min(y)
        y2 = max(y)
        x1,x2,y1,y2=int(x1),int(x2),int(y1),int(y2)

        # 用OpenCV的最小矩形转换成-90到0的角度
        boxes = backward_convert([x[0],y[0],x[1],y[1],x[2],y[2],x[3],y[3]])
        # 根据长短边转成 0 到 180
        new_boxes = coordinate_present_convert(boxes)
        # 转成弧度：
        new_boxes[0][-1] = new_boxes[0][-1] * math.pi/180
        new_boxes = new_boxes.astype(np.float32)
        cx,cy,w,h,angle = new_boxes[0]
        mem[img_name].append([label, x1, y1, x2, y2, cx, cy, w, h, angle])

        with open(os.path.join('/home/kingargroo/seed/roAnnotations', img_name.rstrip('.jpg')) + '.xml', 'w') as f:
            f.write(f"""<annotation>
        <folder>JPEGImages</folder>
        <filename>{img_name}.jpg</filename>
        <size>
            <width>{width}</width>
            <height>{height}</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>\n""")
            for label, x1, y1, x2, y2, cx, cy, w, h, angle in mem[img_name]:
                f.write(f"""<object>
            <type>bndbox</type>
            <name>{label}</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>{x1}</xmin>
                <ymin>{y1}</ymin>
                <xmax>{x2}</xmax>
                <ymax>{y2}</ymax>
            </bndbox>
        </object><object>
        <type>robndbox</type>
        <name>{label}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <robndbox>
        <cx>{cx}</cx>
        <cy>{cy}</cy>
        <w>{w}</w>
        <h>{h}</h>
        <angle>{angle}</angle>
        </robndbox>
        </object>\n""")
            f.write("</annotation>")