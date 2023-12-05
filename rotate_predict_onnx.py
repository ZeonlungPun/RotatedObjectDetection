import colorsys
import numpy as np
import time
import onnxruntime
import cv2


def resize_image(image, size, letterbox_image):
    ih, iw = image.shape[:2]
    h, w = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        new_image = 128 * np.ones((h, w, 3), dtype=np.uint8)
        new_image[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image
    else:
        new_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    return new_image


def preprocess_input(image):
    image /= 255.0
    return image


class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 6 + num_classes
        self.input_shape = input_shape
        # -----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[142, 110],[192, 243],[459, 401]
        #   26x26的特征层对应的anchor是[36, 75],[76, 55],[72, 146]
        #   52x52的特征层对应的anchor是[12, 16],[19, 36],[40, 28]
        # -----------------------------------------------------------#
        self.anchors_mask = anchors_mask

    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            # -----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size = 1
            #   batch_size, 3 * (5 + 1 + 80), 20, 20
            #   batch_size, 255, 40, 40
            #   batch_size, 255, 80, 80
            # -----------------------------------------------#
            batch_size = input.shape[0]
            input_height = input.shape[2]
            input_width = input.shape[3]

            # -----------------------------------------------#
            #   输入为640x640时
            #   stride_h = stride_w = 32、16、8
            # -----------------------------------------------#
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            # -------------------------------------------------#
            #   此时获得的scaled_anchors大小是相对于特征层的
            # -------------------------------------------------#
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                              self.anchors[self.anchors_mask[i]]]

            # -----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 3, 20, 20, 85
            #   batch_size, 3, 40, 40, 85
            #   batch_size, 3, 80, 80, 85
            # -----------------------------------------------#
            prediction = input.reshape(batch_size, len(self.anchors_mask[i]), self.bbox_attrs, input_height,
                                       input_width)
            prediction = np.transpose(prediction, (0, 1, 3, 4, 2))
            # -----------------------------------------------#
            #   先验框的中心位置的调整参数
            # -----------------------------------------------#
            x = 1 / (1 + np.exp(-prediction[..., 0]))
            y = 1 / (1 + np.exp(-prediction[..., 1]))
            # -----------------------------------------------#
            #   先验框的宽高调整参数
            # -----------------------------------------------#
            w = 1 / (1 + np.exp(-prediction[..., 2]))
            h = 1 / (1 + np.exp(-prediction[..., 3]))
            # -----------------------------------------------#
            #   获取旋转角度
            # -----------------------------------------------#
            angle = 1 / (1 + np.exp(-prediction[..., 4]))
            # -----------------------------------------------#
            #   获得置信度，是否有物体
            # -----------------------------------------------#
            conf = 1 / (1 + np.exp(-prediction[..., 5]))
            # -----------------------------------------------#
            #   种类置信度
            # -----------------------------------------------#
            pred_cls = 1 / (1 + np.exp(-prediction[..., 6:]))

            # ----------------------------------------------------------#
            #   生成网格，先验框中心，网格左上角
            #   batch_size,3,20,20
            # ----------------------------------------------------------#
            grid_x = np.linspace(0, input_width - 1, input_width)
            grid_x = np.tile(grid_x, (input_height, 1))
            grid_x = np.tile(grid_x, (batch_size * len(self.anchors_mask[i]), 1, 1)).reshape(x.shape)

            grid_y = np.linspace(0, input_height - 1, input_height)
            grid_y = np.tile(grid_y, (input_width, 1)).T
            grid_y = np.tile(grid_y, (batch_size * len(self.anchors_mask[i]), 1, 1)).reshape(y.shape)

            scaled_anchors = np.array(scaled_anchors)
            anchor_w = scaled_anchors[:, 0:1]
            anchor_h = scaled_anchors[:, 1:2]
            anchor_w = np.tile(anchor_w, (batch_size, 1)).reshape(1, -1, 1)
            anchor_w = np.tile(anchor_w, (1, 1, input_height * input_width)).reshape(w.shape)
            anchor_h = np.tile(anchor_h, (batch_size, 1)).reshape(1, -1, 1)
            anchor_h = np.tile(anchor_h, (1, 1, input_height * input_width)).reshape(h.shape)

            # ----------------------------------------------------------#
            #   利用预测结果对先验框进行调整
            #   首先调整先验框的中心，从先验框中心向右下角偏移
            #   再调整先验框的宽高。
            #   x 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   y 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   w 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            #   h 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            # ----------------------------------------------------------#
            pred_boxes = np.zeros(prediction[..., :4].shape, dtype='float32')
            pred_boxes[..., 0] = x * 2. - 0.5 + grid_x
            pred_boxes[..., 1] = y * 2. - 0.5 + grid_y
            pred_boxes[..., 2] = (w * 2) ** 2 * anchor_w
            pred_boxes[..., 3] = (h * 2) ** 2 * anchor_h
            pred_theta = (angle - 0.5) * np.pi

            # ----------------------------------------------------------#
            #   将输出结果归一化成小数的形式
            # ----------------------------------------------------------#
            _scale = np.array([input_width, input_height, input_width, input_height]).astype('float32')
            output = np.concatenate(
                (pred_boxes.reshape(batch_size, -1, 4) / _scale, pred_theta.reshape(batch_size, -1, 1),
                 conf.reshape(batch_size, -1, 1), pred_cls.reshape(batch_size, -1, self.num_classes)), -1)


            outputs.append(output)
        return outputs

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5,
                            nms_thres=0.4):
        # ----------------------------------------------------------#
        #   prediction  [batch_size, num_anchors, 85]
        # ----------------------------------------------------------#

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # ----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            # ----------------------------------------------------------#
            class_conf = np.max(image_pred[:, 6:6 + num_classes], axis=1, keepdims=True)
            class_pred = np.argmax(image_pred[:, 6:6 + num_classes], axis=1)
            class_pred = np.expand_dims(class_pred, axis=1)

            # ----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            # ----------------------------------------------------------#
            conf_mask = (image_pred[:, 5] * class_conf[:, 0] >= conf_thres).squeeze()
            # ----------------------------------------------------------#
            #   根据置信度进行预测结果的筛选
            # ----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.shape[0]:
                continue
            # -------------------------------------------------------------------------#
            #   detections  [num_anchors, 8]
            #   8的内容为：x, y, w, h, angle, obj_conf, class_conf, class_pred
            # -------------------------------------------------------------------------#
            detections = np.concatenate((image_pred[:, :6], class_conf, class_pred), 1)

            # ------------------------------------------#
            #   获得预测结果中包含的所有种类
            # ------------------------------------------#
            unique_labels = np.unique(detections[:, -1])

            for c in unique_labels:
                # ------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                # ------------------------------------------#
                detections_class = detections[detections[:, -1] == c]

                # ------------------------------------------#
                #   使用cv2.dnn.NMSBoxesRotated进行非极大抑制
                # ------------------------------------------#
                bboxes = [[[bbox[0], bbox[1]], [bbox[2], bbox[3]], bbox[4] * 180 / np.pi] for bbox in
                          detections_class[:, :5]]
                scores = [float(score) for score in detections_class[:, 5] * detections_class[:, 6]]
                indices = cv2.dnn.NMSBoxesRotated(bboxes, scores, conf_thres, nms_thres)
                max_detections = detections_class[indices.flatten()]
                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else np.concatenate((output[i], max_detections))

            if output[i] is not None:
                output[i][:, :5] = self.yolo_correct_boxes(output[i], input_shape, image_shape, letterbox_image)
        return output

    def yolo_correct_boxes(self, output, input_shape, image_shape, letterbox_image):
        # -----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        # -----------------------------------------------------------------#
        box_xy = output[..., 0:2]
        box_wh = output[..., 2:4]
        angle = output[..., 4:5]
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            # -----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            # -----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_xy = box_yx[:, ::-1]
        box_hw = box_wh[:, ::-1]

        rboxes = np.concatenate([box_xy, box_wh, angle], axis=-1)
        rboxes[:, [0, 2]] *= image_shape[1]
        rboxes[:, [1, 3]] *= image_shape[0]
        return rboxes


class YOLO(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path": 'model_data/models.onnx',
        # ---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        # ---------------------------------------------------------------------#
        "input_shape": [1280, 1280],
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.3,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

            # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names = ['Car']
        self.num_classes = 1
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.anchors = np.array([[12, 16], [19, 36], [40, 28],
                                 [36, 75], [76, 55], [72, 146],
                                 [142, 110], [192, 243], [459, 401]])
        self.num_anchors = 9
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                   self.anchors_mask)
        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self):
        # ---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        # ---------------------------------------------------#
        self.net = onnxruntime.InferenceSession(self.model_path,
                                                providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                                           'CPUExecutionProvider'])
        self.output_name = [i.name for i in self.net.get_outputs()]
        self.input_name = [i.name for i in self.net.get_inputs()]

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), True)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        #   h, w, 3 => 3, h, w => 1, 3, h, w
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        # ---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        # ---------------------------------------------------------#
        outputs = self.net.run(self.output_name, {self.input_name[0]: image_data})
        outputs = self.bbox_util.decode_box(outputs)
        # ---------------------------------------------------------#
        #   将预测框进行堆叠，然后进行非极大抑制
        # ---------------------------------------------------------#
        results = self.bbox_util.non_max_suppression(np.concatenate(outputs, axis=1), self.num_classes,
                                                     self.input_shape,
                                                     image_shape, True, conf_thres=self.confidence,
                                                     nms_thres=self.nms_iou)

        if results[0] is None:
            return image

        top_label = np.array(results[0][:, 7], dtype='int32')
        top_conf = results[0][:, 5] * results[0][:, 6]
        top_rboxes = results[0][:, :5]

        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            #predicted_class = self.class_names[int(c)]
            rbox = top_rboxes[i]
            score = top_conf[i]
            rbox = ((rbox[0], rbox[1]), (rbox[2], rbox[3]), rbox[4] * 180 / np.pi)
            cx,cy=rbox[0]
            cv2.circle(image,(int(cx),int(cy)),1,(0, 255, 0),thickness=2)

            poly = cv2.boxPoints(rbox).astype(np.int32)
            x, y = np.min(poly[:, 0]), np.min(poly[:, 1]) - 20
            cv2.polylines(image, [poly.reshape((-1, 1, 2))], True, (255, 0, 0), thickness=2)
            label = ' {:.2f}'.format(score)
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        # text = 'number:{}'.format(len(top_label))
        # imgh, imgw = image.shape[0:2]
        # newh = imgh + 350
        # shape = (newh, imgw, 3)  # y, x, RGB
        # new_img = np.full(shape, 255)
        # new_img[0:imgh, 0:imgw, :] = image.copy()
        # new_img = cv2.putText(new_img.astype(np.int32), text, (15, newh - 230), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 0, 0),
        #                       10)
        # new_img = cv2.putText(new_img.astype(np.int32), 'id:{}'.format(1), (imgw - 500, newh - 230),
        #                       cv2.FONT_HERSHEY_COMPLEX, 5, (0, 0, 0), 10)

        #new_img=new_img[:,:,::-1]
        return image


if __name__ == '__main__':
    yolo = YOLO()

    dir_origin_path = "/home/kingargroo/yolov7-obb/ricetest"
    dir_save_path = "/home/kingargroo/yolov7-obb/img"


    import os

    from tqdm import tqdm

    img_names = os.listdir(dir_origin_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(dir_origin_path, img_name)
            image = cv2.imread(image_path)
            r_image = yolo.detect_image(image)
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            save_path=os.path.join(dir_save_path, img_name.replace(".jpg", ".png"))
            cv2.imwrite(save_path,r_image)

