# 文件名称   ：roxml_to_dota.py
# 功能描述   ：把rolabelimg标注的xml文件转换成dota能识别的xml文件，
#             再转换成dota格式的txt文件
#            把旋转框 cx,cy,w,h,angle，或者矩形框cx,cy,w,h,转换成四点坐标x1,y1,x2,y2,x3,y3,x4,y4
import os
import xml.etree.ElementTree as ET
import math,tqdm,cv2
from pathlib import Path



def convert_dota_to_yolo_obb(dota_root_path: str):
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
                - images
                    - train
                    - val
                - labels
                    - train_original
                    - val_original

        After the function execution, the new labels will be saved in:
            - DOTA
                - labels
                    - train
                    - val
    """
    dota_root_path = Path(dota_root_path)

    # Class names to indices mapping
    class_mapping = {
        'rice': 0,}

    def convert_label(image_name, image_width, image_height, orig_label_dir, save_dir):
        """Converts a single image's DOTA annotation to YOLO OBB format and saves it to a specified directory."""
        orig_label_path = orig_label_dir / f'{image_name}.txt'
        save_path = save_dir / f'{image_name}.txt'

        with orig_label_path.open('r') as f, save_path.open('w') as g:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                class_name = parts[8]
                class_idx = class_mapping[class_name]
                coords = [float(p) for p in parts[:8]]
                normalized_coords = [
                    coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)]
                formatted_coords = ['{:.6g}'.format(coord) for coord in normalized_coords]
                g.write(f"{class_idx} {' '.join(formatted_coords)}\n")

    for phase in ['train', 'val']:
        image_dir = dota_root_path / 'images' / phase
        orig_label_dir = dota_root_path / 'labels' / f'{phase}_original'
        save_dir = dota_root_path / 'labels' / phase

        save_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(image_dir.iterdir())
        for image_path in tqdm.tqdm(image_paths, desc=f'Processing {phase} images'):
            if image_path.suffix != '.jpg':
                continue
            image_name_without_ext = image_path.stem
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir)





if __name__ == '__main__':

    convert_dota_to_yolo_obb(dota_root_path='/home/kingargroo/seed/dota')



