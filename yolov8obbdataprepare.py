# name   ：roxml_to_dota.py
# function: transform xml file labeled by rolabelimg to dota xml file
#             then transform to txt file with dota format
#            obb bounding box cx,cy,w,h,angle，or cx,cy,w,h, to four points coordinate x1,y1,x2,y2,x3,y3,x4,y4
import xml.etree.ElementTree as ET
import cv2,os,random,shutil,math
from pathlib import Path
from ultralytics.utils import LOGGER, TQDM

#First step:rolabelImg file to dota xml file to dota txt file

def edit_xml(xml_file, dotaxml_file):
    """
    :param xml_file
    :return:
    """
    tree = ET.parse(xml_file)
    objs = tree.findall('object')
    for ix, obj in enumerate(objs):
        x0 = ET.Element("x0")  # create node
        y0 = ET.Element("y0")
        x1 = ET.Element("x1")
        y1 = ET.Element("y1")
        x2 = ET.Element("x2")
        y2 = ET.Element("y2")
        x3 = ET.Element("x3")
        y3 = ET.Element("y3")


        if (obj.find('robndbox') == None):
            obj_bnd = obj.find('bndbox')
            obj_xmin = obj_bnd.find('xmin')
            obj_ymin = obj_bnd.find('ymin')
            obj_xmax = obj_bnd.find('xmax')
            obj_ymax = obj_bnd.find('ymax')
            #以防有负值坐标
            xmin = max(float(obj_xmin.text),0)
            ymin = max(float(obj_ymin.text),0)
            xmax = max(float(obj_xmax.text),0)
            ymax = max(float(obj_ymax.text),0)
            obj_bnd.remove(obj_xmin)  # 删除节点
            obj_bnd.remove(obj_ymin)
            obj_bnd.remove(obj_xmax)
            obj_bnd.remove(obj_ymax)
            x0.text = str(xmin)
            y0.text = str(ymax)
            x1.text = str(xmax)
            y1.text = str(ymax)
            x2.text = str(xmax)
            y2.text = str(ymin)
            x3.text = str(xmin)
            y3.text = str(ymin)
        else:
            obj_bnd = obj.find('robndbox')
            obj_bnd.tag = 'bndbox'  #
            obj_cx = obj_bnd.find('cx')
            obj_cy = obj_bnd.find('cy')
            obj_w = obj_bnd.find('w')
            obj_h = obj_bnd.find('h')
            obj_angle = obj_bnd.find('angle')
            cx = float(obj_cx.text)
            cy = float(obj_cy.text)
            w = float(obj_w.text)
            h = float(obj_h.text)
            angle = float(obj_angle.text)
            obj_bnd.remove(obj_cx)  # delete node
            obj_bnd.remove(obj_cy)
            obj_bnd.remove(obj_w)
            obj_bnd.remove(obj_h)
            obj_bnd.remove(obj_angle)

            x0.text, y0.text = rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
            x1.text, y1.text = rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
            x2.text, y2.text = rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
            x3.text, y3.text = rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)

        obj_bnd.append(x0)  # add new note
        obj_bnd.append(y0)
        obj_bnd.append(x1)
        obj_bnd.append(y1)
        obj_bnd.append(x2)
        obj_bnd.append(y2)
        obj_bnd.append(x3)
        obj_bnd.append(y3)

        tree.write(dotaxml_file, method='xml', encoding='utf-8')  # update xml file


# to four points coordinate
def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp - xc
    yoff = yp - yc
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return str(int(xc + pResx)), str(int(yc + pResy))



def totxt(xml_path, out_path):
    files = os.listdir(xml_path)
    for file in files:

        tree = ET.parse(xml_path + os.sep + file)
        root = tree.getroot()

        name = file.strip('.xml')
        output = out_path + name + '.txt'
        file = open(output, 'w')

        objs = tree.findall('object')
        for obj in objs:
            cls = obj.find('name').text
            box = obj.find('bndbox')
            x0 = int(float(box.find('x0').text))
            y0 = int(float(box.find('y0').text))
            x1 = int(float(box.find('x1').text))
            y1 = int(float(box.find('y1').text))
            x2 = int(float(box.find('x2').text))
            y2 = int(float(box.find('y2').text))
            x3 = int(float(box.find('x3').text))
            y3 = int(float(box.find('y3').text))
            file.write("{} {} {} {} {} {} {} {} {} 0\n".format(x0, y0, x1, y1, x2, y2, x3, y3, "rice"))
        file.close()
        print(output)

# Second step: split the data to train set and label set
def SplitDataForDota(train_ratio,newdataset_root_file,img_root_file,label_root_file):
    """
     The directory structure assumed for the DOTA dataset:
            - DOTA
                ├─ images
                │   ├─ train
                │   └─ val
                └─ labels
                    ├─ train_original
                    └─ val_original
    :param train_ratio:
    :param newdataset_root_file:
    :param img_root_file:
    :param label_root_file:
    :return:
    """
    # create new files
    os.makedirs(newdataset_root_file + '/images')
    os.makedirs(newdataset_root_file + '/labels')
    os.makedirs(newdataset_root_file + '/images' + '/train')
    os.makedirs(newdataset_root_file + '/images' + '/val')
    os.makedirs(newdataset_root_file + '/labels' + '/train_original')
    os.makedirs(newdataset_root_file + '/labels' + '/val_original')

    # spilt the iamge and labels according to idx
    img_list = os.listdir(img_root_file)
    label_list = os.listdir(label_root_file)
    img_num = len(img_list)
    label_num = len(label_list)
    assert img_num == label_num
    random_numbers = [random.randint(0, img_num - 1) for _ in range(img_num)]
    train_num = int(train_ratio * img_num)
    train_id = random_numbers[:train_num]
    test_id = random_numbers[train_num::]

    train_img_list, test_img_list = [], []
    train_label_list, test_label_list = [], []

    for id in train_id:
        train_img_list.append(img_list[id])
        name = img_list[id].split(".")[0]
        label_name = name + '.txt'
        train_label_list.append(label_name)
    for id_ in test_id:
        test_img_list.append(img_list[id_])
        name = img_list[id_].split(".")[0]
        label_name = name + '.txt'
        test_label_list.append(label_name)

    assert len(train_img_list) == len(train_label_list)
    assert len(test_img_list) == len(test_label_list)

    for img_name, label_name in zip(train_img_list, train_label_list):
        img_raw_path = os.path.join(img_root_file, img_name)
        label_raw_path = os.path.join(label_root_file, label_name)
        shutil.copy(img_raw_path, newdataset_root_file + '/images/train')
        shutil.copy(label_raw_path, newdataset_root_file + '/labels/train_original')

    for img_name, label_name in zip(test_img_list, test_label_list):
        img_raw_path = os.path.join(img_root_file, img_name)
        label_raw_path = os.path.join(label_root_file, label_name)
        shutil.copy(img_raw_path, newdataset_root_file + '/images/val')
        shutil.copy(label_raw_path, newdataset_root_file + '/labels/val_original')






# third step: transform the dota to yolov8 format



def convert_dota_to_yolo_obb(dota_root_path: str,class_mapping):
    """
    Converts DOTA dataset annotations to YOLO OBB (Oriented Bounding Box) format.

    The function processes images in the 'train' and 'val' folders of the DOTA dataset. For each image, it reads the
    associated label from the original labels directory and writes new labels in YOLO OBB format to a new directory.

    Args:
        dota_root_path (str): The root directory path of the DOTA dataset.

    Example:
        ```python
        from ultralytics.data.converter import convert_dota_to_yolo_obb

        convert_dota_to_yolo_obb('path/to/DOTA')
        ```

    Notes:
        The directory structure assumed for the DOTA dataset:
            - DOTA
                ├─ images
                │   ├─ train
                │   └─ val
                └─ labels
                    ├─ train_original
                    └─ val_original

        After execution, the function will organize the labels into:
            - DOTA
                └─ labels
                    ├─ train
                    └─ val
    """
    dota_root_path = Path(dota_root_path)


    # Class names to indices mapping
    # class_mapping = {
    #     "plane": 0,
    #     "ship": 1,
    #     "storage-tank": 2,
    #     "baseball-diamond": 3,
    #     "tennis-court": 4,
    #     "basketball-court": 5,
    #     "ground-track-field": 6,
    #     "harbor": 7,
    #     "bridge": 8,
    #     "large-vehicle": 9,
    #     "small-vehicle": 10,
    #     "helicopter": 11,
    #     "roundabout": 12,
    #     "soccer-ball-field": 13,
    #     "swimming-pool": 14,
    #     "container-crane": 15,
    #     "airport": 16,
    #     "helipad": 17,
    # }

    def convert_label(image_name, image_width, image_height, orig_label_dir, save_dir):
        """Converts a single image's DOTA annotation to YOLO OBB format and saves it to a specified directory."""
        orig_label_path = orig_label_dir / f"{image_name}.txt"
        save_path = save_dir / f"{image_name}.txt"

        with orig_label_path.open("r") as f, save_path.open("w") as g:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                class_name = parts[8]
                print(image_name)
                class_idx = class_mapping[class_name]
                coords = [float(p) for p in parts[:8]]
                normalized_coords = [
                    coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)
                ]
                formatted_coords = ["{:.6g}".format(coord) for coord in normalized_coords]
                g.write(f"{class_idx} {' '.join(formatted_coords)}\n")

    for phase in ["train", "val"]:
        image_dir = dota_root_path / "images" / phase
        orig_label_dir = dota_root_path / "labels" / f"{phase}_original"
        save_dir = dota_root_path / "labels" / phase

        save_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(image_dir.iterdir())
        for image_path in TQDM(image_paths, desc=f"Processing {phase} images"):
            #if image_path.suffix != ".png":
            #    continue
            image_name_without_ext = image_path.stem
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir)


if __name__ == '__main__':
    dotaxml_path = r'/home/kingargroo/seed/ricelabel/label2'  #
    out_path = r'/home/kingargroo/seed/ricelabel/label3/'
    #totxt(dotaxml_path, out_path)
    train_ratio=0.85
    newdataset_root_file = '/home/kingargroo/seed/yolov8'
    img_root_file = '/home/kingargroo/seed/rice'
    label_root_file = '/home/kingargroo/seed/ricelabel/label3'
    SplitDataForDota(train_ratio, newdataset_root_file, img_root_file, label_root_file)
    convert_dota_to_yolo_obb("/home/kingargroo/seed/yolov8",class_mapping={"rice":0})