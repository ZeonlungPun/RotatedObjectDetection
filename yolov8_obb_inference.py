from ultralytics import YOLO
import cv2,math
import numpy as np

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
    polys = np.concatenate([[point1], [point2], [point3], [point4]], axis=0)
    polys=np.array(polys,np.int32)
    #polys = get_best_begin_point(polys)
    return polys

model = YOLO('/home/kingargroo/YOLOVISION/yolov8s-obb.pt')  # load a pretrained model (recommended for training)

# Train the model
img_name='/home/kingargroo/YOLOVISION/test4.jpg'
img=cv2.imread(img_name)
results = model(img_name,stream=True)
for result in results:
    boxes=result.obb.xywhr
    for box in boxes:
        cx, cy, w, h,angle=box[0],box[1],box[2],box[3],box[4]
        cx, cy, w, h, angle =cx.cpu().numpy(),cy.cpu().numpy(),w.cpu().numpy(),h.cpu().numpy(),angle.cpu().numpy()
        # angle in [-pi/4,3/4 pi) --》 [-pi/2,pi/2)
        if angle >= math.pi/2 and angle < 0.75*math.pi:
            angle=angle-math.pi
        polys=rbox2poly(np.array([cx,cy,w,h,angle]))
        cv2.polylines(img, [polys], isClosed=True, color=(0, 0, 255), thickness=1)

cv2.imwrite('/home/kingargroo/YOLOVISION/test11.jpg',img)
        # #  [-pi/2,pi/2) --> [-pi/2,0)
        # if angle>=-math.pi/2 and angle < 0:
        #     continue
        # else:
        #     angle=angle-math.pi/2
        #     temp1,temp2=w,h
        #     w=temp2
        #     h=temp1
        # # [-pi/2,0) --> (-pi/2,0]
        # if angle== -math.pi/2:
        #     angle=0
