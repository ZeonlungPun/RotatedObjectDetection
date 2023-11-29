import os,random
import numpy as np
from PIL import Image,ImageDraw
import cv2,math
import xml.etree.ElementTree as ET

def save_dota_xml(img_name,rbox,class_name,dest_file,img_size):

    h,w=img_size
    c=3
    annotation_xml=ET.Element('annotation')
    folder_xml=ET.SubElement(annotation_xml,'folder')
    folder_xml.text='rice'
    filename_xml=ET.SubElement(annotation_xml,'filename')
    filename_xml.text=img_name
    path_xml=ET.SubElement(annotation_xml,'path')
    path_xml.text=img_name
    size_xml=ET.SubElement(annotation_xml,'size')
    width_xml=ET.SubElement(size_xml,'width')
    height_xml = ET.SubElement(size_xml, 'height')
    depth_xml=ET.SubElement(size_xml,'depth')
    width_xml.text,height_xml.text,depth_xml.text=str(w),str(h),str(c)

    for i in range(len(rbox)):
        box=rbox[i]
        x0, y0 ,x1 ,y1 ,x2 ,y2 ,x3 ,y3,cls=int(box[0]),int(box[1]), int(box[2]),int(box[3]),int(box[4]),int(box[5]),int(box[6]),int(box[7]),int(box[8])
        object_xml=ET.SubElement(annotation_xml,'object')
        type_xml=ET.SubElement(object_xml,'type')
        type_xml.text='robndbox'
        name_xml=ET.SubElement(object_xml,'name')
        name_xml.text=class_name
        pose_xml=ET.SubElement(object_xml,'pose')
        pose_xml.text='Unspecified'
        truncated_xml=ET.SubElement(object_xml,'truncated')
        truncated_xml.text='0'
        difficult_xml=ET.SubElement(object_xml,'difficult')
        difficult_xml.text='0'
        bndbox_xml=ET.SubElement(object_xml,'bndbox')
        x0_xml,y0_xml,x1_xml,y1_xml,x2_xml,y2_xml,x3_xml,y3_xml=ET.SubElement(bndbox_xml,'x0'),ET.SubElement(bndbox_xml,'y0'),ET.SubElement(bndbox_xml,'x1'),ET.SubElement(bndbox_xml,'y1'),ET.SubElement(bndbox_xml,'x2'),ET.SubElement(bndbox_xml,'y2'),ET.SubElement(bndbox_xml,'x3'),ET.SubElement(bndbox_xml,'y3')
        x0_xml.text, y0_xml.text, x1_xml.text, y1_xml.text, x2_xml.text, y2_xml.text, x3_xml.text, y3_xml.text =str(x0), str(y0) ,str(x1) ,str(y1) ,str(x2) ,str(y2) ,str(x3) ,str(y3)
    tree=ET.ElementTree(annotation_xml)
    xml_dest_file=os.path.join(dest_file,img_name.split('.')[0])+'.xml'
    tree.write(xml_dest_file, method='xml', encoding='utf-8')  # update xml file

def cal_line_length(point1, point2):
    """Calculate the length of line.
    Args:
        point1 (List): [x,y]
        point2 (List): [x,y]
    Returns:
        length (float)
    """
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) +
        math.pow(point1[1] - point2[1], 2))
def get_best_begin_point_single(coordinate):
    """Get the best begin point of the single polygon.
    Args:
        coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    Returns:
        reorder coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
               [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
               [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
               [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combine[i][0], dst_coordinate[0]) \
                     + cal_line_length(combine[i][1], dst_coordinate[1]) \
                     + cal_line_length(combine[i][2], dst_coordinate[2]) \
                     + cal_line_length(combine[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.hstack(
        (np.array(combine[force_flag]).reshape(8)))
def poly2rbox(polys):
    """
    Trans poly format to rbox format.
    Args:
        polys (array): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4])
    Returns:
        rboxes (array): (num_gts, [cx cy l s θ])
    """
    assert polys.shape[-1] == 8
    rboxes = []
    for poly in polys:
        poly = np.float32(poly.reshape(4, 2))
        (x, y), (w, h), angle = cv2.minAreaRect(poly) # θ ∈ [0， 90]
        theta = angle / 180 * np.pi #angle to rad systemd
        # trans opencv format to longedge format θ ∈ [-pi/2， pi/2]
        if w < h:
            w, h = h, w
            theta += np.pi / 2
        while not np.pi / 2 > theta >= -np.pi / 2:
            if theta >= np.pi / 2:
                theta -= np.pi
            else:
                theta += np.pi
        assert np.pi / 2 > theta >= -np.pi / 2
        rboxes.append([x, y, w, h, theta])
    return np.array(rboxes)
def rbox2poly(obboxes):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    try:
        center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
    except:
        results = np.stack([0., 0., 0., 0., 0., 0., 0., 0.], axis=-1)
        return results.reshape(1, -1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    polys = np.concatenate([point1, point2, point3, point4], axis=-1)
    polys = get_best_begin_point(polys)
    return polys


def get_best_begin_point(coordinates):
    """Get the best begin points of polygons.
    Args:
        coordinate (ndarray): shape(n, 8).
    Returns:
        reorder coordinate (ndarray): shape(n, 8).
    """
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def rand( a=0, b=1):
    return np.random.rand()*(b-a) + a


def merge_rboxes( rboxes, cutx, cuty):
    merge_rbox = []
    for i in range(len(rboxes)):
        for rbox in rboxes[i]:
            tmp_rbox = []
            xc, yc, w, h = rbox[0], rbox[1], rbox[2], rbox[3]
            tmp_rbox.append(xc)
            tmp_rbox.append(yc)
            tmp_rbox.append(h)
            tmp_rbox.append(w)
            tmp_rbox.append(rbox[-1])
            merge_rbox.append(rbox)
    merge_rbox = np.array(merge_rbox)
    return merge_rbox

def get_random_data(annotation_line, input_shape,img_main_path='/home/kingargroo/seed/rice',jitter=.3, hue=.1, sat=0.7, val=0.4, random=True,show=False):
    img_subpath = annotation_line[0]
    boxes = annotation_line[1::]
    img_path = os.path.join(img_main_path, img_subpath) + '.jpg'
    image = Image.open(img_path)
    image = cvtColor(image)
    # ------------------------------#
    #   get the height and width of image
    # ------------------------------#
    iw, ih = image.size
    h, w = input_shape
    # ------------------------------#
    #   get the box
    # ------------------------------#
    box = np.array([np.array(box_) for box_ in boxes])

    if not random:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        #   padding the gray in extra part
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)


        #   adjust the bounding box correspon to image
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2, 4, 6]] = box[:, [0, 2, 4, 6]] * nw / iw + dx
            box[:, [1, 3, 5, 7]] = box[:, [1, 3, 5, 7]] * nh / ih + dy

            #   polygon to rbox
            rbox = np.zeros((box.shape[0], 6))
            rbox[..., :5] = poly2rbox(box[..., :8])
            rbox[..., 5] = box[..., 8]
            keep = (rbox[:, 0] >= 0) & (rbox[:, 0] < w) \
                   & (rbox[:, 1] >= 0) & (rbox[:, 0] < h) \
                   & (rbox[:, 2] > 5) | (rbox[:, 3] > 5)
            rbox = rbox[keep]
        return image_data, rbox


    #   rescale the img and twist the width and height
    # ------------------------------------------#
    new_ar = iw / ih * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)


    #   padding the gray in extra part
    # ------------------------------------------#
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    #   flip the image
    # ------------------------------------------#
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    image_data = np.array(image, np.uint8)

    #   HSV augment
    #   calculate the parameter of   HSV augment
    # ---------------------------------#
    r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

    #  rgb to hsv space
    # ---------------------------------#
    hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
    dtype = image_data.dtype

    #   apply the change
    # ---------------------------------#
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

    #   adjust the bounding box
    # ---------------------------------#
    if len(box) > 0:
        np.random.shuffle(box)
        # padding the gray in extra part,need to adjust correspondingly
        box[:, [0, 2, 4, 6]] = box[:, [0, 2, 4, 6]] * nw / iw + dx
        box[:, [1, 3, 5, 7]] = box[:, [1, 3, 5, 7]] * nh / ih + dy
        if flip: box[:, [0, 2, 4, 6]] = w - box[:, [0, 2, 4, 6]]

        #  polygon to rbox
        rbox = np.zeros((box.shape[0], 6))
        rbox[..., :5] = poly2rbox(box[..., :8])
        rbox[..., 5] = box[..., 8]
        keep = (rbox[:, 0] >= 0) & (rbox[:, 0] < w) \
               & (rbox[:, 1] >= 0) & (rbox[:, 0] < h) \
               & (rbox[:, 2] > 5) | (rbox[:, 3] > 5)
        rbox = rbox[keep]

    # check the image#
    if show:
        draw = ImageDraw.Draw(image)
        polys = rbox2poly(rbox[..., :5])
        for poly in polys:
            draw.polygon(xy=list(poly))
        image.show()
    return image_data, rbox

def get_random_data_with_MixUp(image_1, rbox_1, image_2, rbox_2,show=False):
    new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
    if len(rbox_1) == 0:
        new_rboxes = rbox_2
    elif len(rbox_2) == 0:
        new_rboxes = rbox_1
    else:
        new_rboxes = np.concatenate([rbox_1, rbox_2], axis=0)

    if show:
        new_img = Image.fromarray(np.uint8(new_image))
        draw = ImageDraw.Draw(new_img)
        polys = rbox2poly(new_rboxes[..., :5])
        for poly in polys:
            draw.polygon(xy=list(poly))
        new_img.show()
    return new_image, new_rboxes



def get_random_data_with_Mosaic( annotation_line, input_shape,img_main_path='/home/kingargroo/seed/rice',jitter=0.3, hue=.1, sat=0.7, val=0.4, show=True):
    """four image combine to 1 """
    h, w = input_shape
    min_offset_x = rand(0.3, 0.7)
    min_offset_y = rand(0.3, 0.7)

    image_datas = []
    rbox_datas = []
    index = 0
    for line in annotation_line:
        # ---------------------------------#
        # get first image information
        # ---------------------------------#
        img_subpath=line[0]
        boxes=line[1::]
        img_path=os.path.join(img_main_path,img_subpath)+'.jpg'
        # ---------------------------------#
        #   read the image
        # ---------------------------------#
        image = Image.open(img_path)
        image = cvtColor(image)

        # ---------------------------------#
        #   get image size
        # ---------------------------------#
        iw, ih = image.size
        # ---------------------------------#
        #   bounding box
        # ---------------------------------#
        box = np.array([np.array(box_) for box_ in boxes])
        # ---------------------------------#
        #   filp the image or not
        # ---------------------------------#
        flip = rand() < .5
        if flip and len(box) > 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            box[:, [0, 2, 4, 6]] = iw - box[:, [0, 2, 4, 6]]

        #   rescale the img and twist the width and height
        new_ar = iw / ih * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(.4, 1)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)


        # put the four image to new location
        # -----------------------------------------------#
        if index == 0:
            dx = int(w * min_offset_x) - nw
            dy = int(h * min_offset_y) - nh
        elif index == 1:
            dx = int(w * min_offset_x) - nw
            dy = int(h * min_offset_y)
        elif index == 2:
            dx = int(w * min_offset_x)
            dy = int(h * min_offset_y)
        elif index == 3:
            dx = int(w * min_offset_x)
            dy = int(h * min_offset_y) - nh

        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image)

        index = index + 1
        rbox_data = []

        #   deal with the rbox
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2, 4, 6]] = box[:, [0, 2, 4, 6]] * nw / iw + dx
            box[:, [1, 3, 5, 7]] = box[:, [1, 3, 5, 7]] * nh / ih + dy
            # ------------------------------#
            #   polygon to rbox
            # ------------------------------#
            rbox = np.zeros((box.shape[0], 6))
            rbox[..., :5] = poly2rbox(box[..., :8])
            rbox[..., 5] = box[..., 8]
            keep = (rbox[:, 0] >= 0) & (rbox[:, 0] < w) \
                   & (rbox[:, 1] >= 0) & (rbox[:, 0] < h) \
                   & (rbox[:, 2] > 5) | (rbox[:, 3] > 5)
            rbox = rbox[keep]
            rbox_data = np.zeros((len(rbox), 6))
            rbox_data[:len(rbox)] = rbox

        image_datas.append(image_data)
        rbox_datas.append(rbox_data)

    #   cut the image and paste together
    cutx = int(w * min_offset_x)
    cuty = int(h * min_offset_y)

    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    new_image = np.array(new_image, np.uint8)

    # hsv change
    r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
    dtype = new_image.dtype
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

    new_rboxes = merge_rboxes(rbox_datas, cutx, cuty)

    #check
    if show:
        new_img = Image.fromarray(new_image)
        draw = ImageDraw.Draw(new_img)
        polys = rbox2poly(new_rboxes[..., :5])
        for poly in polys:
            draw.polygon(xy=list(poly))
        new_img.show()
    return new_image, new_rboxes

def GetAugImgAndLabel(annotation_lines,aug_num,aug_path,dest_file):

    for i in range(aug_num):
        input_lines = [annotation_lines[random.randint(0, len(annotation_lines) - 1)] for _ in range(4)]
        new_image, new_box = get_random_data_with_Mosaic(input_lines, input_shape=(1280, 1280), jitter=0.3, hue=.1,
                                                         sat=0.7, val=0.4, show=False)
        save_path=os.path.join(aug_path,str(i))+'.jpg'
        cv2.imwrite(save_path,new_image)
        new_box_=np.zeros((len(new_box),9))
        poly=rbox2poly(new_box[:,:5])
        new_box_[:,:8]=poly
        new_box_[:,8]=new_box[:,5]
        save_dota_xml(img_name=str(i)+'.jpg',rbox=new_box_,class_name='rice',dest_file=dest_file,img_size=(1280, 1280))

    for j in range(aug_num,2*aug_num):
        k1,k2=random.randint(0, len(annotation_lines) - 1),random.randint(0, len(annotation_lines) - 1)
        image_1, rbox_1 = get_random_data(annotation_lines[k1], input_shape=(1280, 1280), random=True)
        image_2, rbox_2 = get_random_data(annotation_lines[k2], input_shape=(1280, 1280), random=True)
        new_image, new_rboxes=get_random_data_with_MixUp(image_1, rbox_1, image_2, rbox_2)
        poly = rbox2poly(new_box[:, :5])
        new_box_[:, :8] = poly
        new_box_[:, 8] = new_box[:, 5]
        save_dota_xml(img_name=str(j)+'.jpg',rbox=new_box_,class_name='rice',dest_file=dest_file,img_size=(1280, 1280))
        save_path = os.path.join(aug_path, str(j)) + '.jpg'
        cv2.imwrite(save_path, new_image)


if __name__=="__main__":
    xml_file_list=os.listdir('/home/kingargroo/seed/rice/label2')
    annotation_lines = []
    for i in xml_file_list:
        xml_file=os.path.join('/home/kingargroo/seed/rice/label2',i)
        try:
            tree = ET.parse(xml_file)
            objs = tree.findall('object')
            annotation_line=[]
            img_name=tree.findall('filename')[0].text
            annotation_line.append(img_name)
            class_id=0
            for ix, obj in enumerate(objs):
                obj_bnd = obj.find('bndbox')
                obj_x0 = float(obj_bnd.find('x0').text)
                obj_y0 = float(obj_bnd.find('y0').text)
                obj_x1 = float(obj_bnd.find('x1').text)
                obj_y1 = float(obj_bnd.find('y1').text)
                obj_x2 = float(obj_bnd.find('x2').text)
                obj_y2 = float(obj_bnd.find('y2').text)
                obj_x3 = float(obj_bnd.find('x3').text)
                obj_y3 = float(obj_bnd.find('y3').text)
                box=[obj_x0,obj_y0,obj_x1,obj_y1,obj_x2,obj_y2,obj_x3,obj_y3,class_id]
                annotation_line.append(box)
            annotation_lines.append(annotation_line)
        except:
            continue
    GetAugImgAndLabel(annotation_lines,aug_num=20,aug_path='/home/kingargroo/seed/rice_aug',dest_file='/home/kingargroo/seed/rice_aug_label')



    #print(len(new_box))