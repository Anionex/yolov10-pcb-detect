"""
VOC 格式的数据集转化为 COCO 格式的数据集
"""

import os
import cv2
import json
from tqdm import tqdm
from random import sample
import xml.etree.ElementTree as ET



class_name = ['Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
with open('PCB_DATASET/classes.txt', 'w') as f:
    for name in class_name:
        f.writelines(name+'\n')
        


def train_test_val_split_random(file_dir, file_list, coco_path,train_percent=0.8,val_percent=0.1,test_percent=0.1):
    # 这里可以修改数据集划分的比例。
    assert int(train_percent+test_percent+val_percent) == 1
    
    num_file = len(file_list)
    print(num_file)
    
    num_train = int(num_file * train_percent)
    num_val = int(num_file * val_percent)
    num_test = num_file - num_train - num_val

    val_file_list = sample(file_list, num_val)
    
    train_dir = os.path.join(coco_path, 'train2017')
    val_dir = os.path.join(coco_path, 'val2017')
    test_dir = os.path.join(coco_path, 'test2017')
    
    class_list = ['train2017','val2017','test2017']
    
    for item in class_list:
        path = os.path.join(coco_path, item)
        if not os.path.exists(path):
            os.mkdir(path)
    
    for i in val_file_list:
        if i.endswith('.jpg') :
            file_path = os.path.join(file_dir, i)
            new_path = os.path.join(coco_path, 'val2017', i)
            mox.file.copy(file_path, new_path)

    var_test_file_list = [i for i in file_list if not i in val_file_list]
    test_file_list = sample(var_test_file_list, num_test)

    for i in test_file_list:
        if i.endswith('.jpg'):
            file_path = os.path.join(file_dir, i)
            new_path = os.path.join(coco_path, 'test2017', i)
            mox.file.copy(file_path, new_path)
    
    train_file_list = []
    for i in file_list:
        if i not in val_file_list and i not in test_file_list:
            if i.endswith('.jpg') :
                file_path = os.path.join(file_dir, i)
                new_path = os.path.join(coco_path, 'train2017', i)
                mox.file.copy(file_path, new_path)
                train_file_list.append(i)
    
    return train_file_list, val_file_list, test_file_list


def get(root, name):
    vars = root.findall(name)
    return vars
 

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

def voc2coco(root_path, coco_path):
    print("Loading data from ",root_path)

    assert os.path.exists(root_path)
    
    if not os.path.exists(coco_path):
        os.makedirs(coco_path)
    
    image_dir = ['images/Mouse_bite', 'images/Open_circuit', 'images/Short', 'images/Spur', 'images/Spurious_copper']
    
    with open(os.path.join(root_path, 'classes.txt')) as f:
        classes = f.read().strip().split()
        
    mox.file.copy(os.path.join(root_path, 'classes.txt'), os.path.join(coco_path, 'classes.txt'))
    
    cur_index = 0
    
    # 标注的id
    ann_id_cnt = 0
    
    # 用于保存所有数据的图片信息和标注信息
    train_dataset = {'categories': [], 'annotations': [], 'images': []}
    val_dataset = {'categories': [], 'annotations': [], 'images': []}
    test_dataset = {'categories': [], 'annotations': [], 'images': []}
    
    # 建立类别标签和数字id的对应关系, 类别id从0开始。
    for i, cls in enumerate(classes, 0):
        train_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
        val_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
        test_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
    
    for item in image_dir:
        originImagesDir = os.path.join(root_path, item)
        originLabelsDir = originImagesDir.replace('images','Annotations') 

        # images dir name
        indexes = os.listdir(originImagesDir)

        print("spliting mode: random split")
        train_img, val_img, test_img = train_test_val_split_random(originImagesDir, indexes, coco_path,0.8,0.1,0.1)

        for k, index in enumerate(tqdm(indexes)):
            if index.endswith('.jpg') :
                xmlFile = index.replace('.jpg','.xml')
                xmlFile_Path = os.path.join(originLabelsDir, xmlFile)
                if not os.path.exists(xmlFile_Path):
                    # 如没标签，跳过，只保留图片信息。
                    continue
                    
                # 读取图像的宽和高
                tree = ET.parse(xmlFile_Path)
                root = tree.getroot()
                size = get_and_check(root, 'size', 1)
                width = int(get_and_check(size, 'width', 1).text)
                height = int(get_and_check(size, 'height', 1).text)
                
                # 切换dataset的引用对象，从而划分数据集
                if index in train_img:
                    dataset = train_dataset
                elif index in val_img:
                    dataset = val_dataset
                elif index in test_img:
                    dataset = test_dataset
                # 添加图像的信息
                dataset['images'].append({'file_name': index,
                                            'id': cur_index,
                                            'width': width,
                                            'height': height})
                
                for obj in get(root, 'object'):
                    category = get_and_check(obj, 'name', 1).text
                    cls_id = len(classes)
                    for i, cls in enumerate(classes):
                        if category.upper() == cls.upper():
                            cls_id = i
                            break
                    bndbox = get_and_check(obj, 'bndbox', 1)
                    x1 = int(get_and_check(bndbox, 'xmin', 1).text) - 1
                    y1 = int(get_and_check(bndbox, 'ymin', 1).text) - 1
                    x2 = int(get_and_check(bndbox, 'xmax', 1).text)
                    y2 = int(get_and_check(bndbox, 'ymax', 1).text)
                    assert (x2 > x1)
                    assert (y2 > y1)
                    width = abs(x2 - x1)
                    height = abs(y2 - y1)

                    dataset['annotations'].append({
                        'area': width * height,
                        'bbox': [x1, y1, width, height],
                        'category_id': cls_id,
                        'id': ann_id_cnt,
                        'image_id': cur_index,
                        'iscrowd': 0,
                        # mask, 矩形是从左上角点按顺时针的四个顶点
                        'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                    })
                    ann_id_cnt += 1

                cur_index = cur_index + 1

    # 保存结果
    folder = os.path.join(coco_path, 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    for phase in ['instances_train2017','instances_val2017','instances_test2017']:
        json_name = os.path.join(coco_path, 'annotations/{}.json'.format(phase))
        with open(json_name, 'w') as f:
            if phase == 'instances_train2017':
                json.dump(train_dataset, f)
            elif phase == 'instances_val2017':
                json.dump(val_dataset, f)
            elif phase == 'instances_test2017':
                json.dump(test_dataset, f)
        print('Save annotation to {}'.format(json_name))

voc2coco('PCB_DATASET', 'PCB_DATASET_COCO')