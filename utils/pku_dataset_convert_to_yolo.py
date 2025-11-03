import os
import random
import re
import shutil

from lxml import etree
from PIL import Image

TEST_RATIO = 1
VAL_RATIO = 0

OUTPUT_ROOT = "./yolo_dataset"

PLAIN_ANNOTATION_PATH = "./Annotations"
PLAIN_IMAGE_PATH = "./images"


def get_subset_path():
    r = random.random()
    if r < VAL_RATIO:
        p = os.path.join(OUTPUT_ROOT, "val")
        os.makedirs(p, exist_ok=True)
        return p
    elif r < VAL_RATIO + TEST_RATIO:
        p = os.path.join(OUTPUT_ROOT, "test")
        os.makedirs(p, exist_ok=True)
        return p
    else:
        p = os.path.join(OUTPUT_ROOT, "train")
        os.makedirs(p, exist_ok=True)
        return p


def handle_plain():
    for p, sd, sf in os.walk(PLAIN_ANNOTATION_PATH):
        pi = os.path.join(PLAIN_IMAGE_PATH, os.path.relpath(p, PLAIN_ANNOTATION_PATH))
        for f in sf:
            f = os.path.splitext(os.path.basename(f))[0]
            from_annotation = os.path.join(p, f + ".xml")
            from_img = os.path.join(pi, f + ".jpg")
            to = get_subset_path()
            shutil.copy(from_annotation, os.path.join(to, f + ".xml"))
            shutil.copy(from_img, os.path.join(to, f + ".jpg"))


MAPPING = {
    "missing_hole": 2,
    "mouse_bite": 0,
    "open_circuit": 4,
    "short": 3,
    "spur": 1,
    "spurious_copper": 5
}


def post_conversion():
    for p, sd, sf in os.walk(OUTPUT_ROOT):
        for f in sf:
            if f.endswith(".xml"):
                fn = os.path.splitext(f)[0]
                pf = fn + ".jpg"

                fp = os.path.join(p, f)
                pfp = os.path.join(p, pf)
                tree = etree.parse(fp)
                image = Image.open(pfp)
                size = tree.getroot().find("size")
                w = int(size.find("width").text)
                h = int(size.find("height").text)
                ts = re.search(r"\d+_(\S+)_\d+", fn).group(1)
                
                with open(os.path.join(p, fn + ".txt"), "w") as writable:
                    for obj in tree.getroot().findall("object"):
                        box = obj.find("bndbox")
                        xmin = int(box.find("xmin").text) / w
                        ymin = int(box.find("ymin").text) / h
                        xmax = int(box.find("xmax").text) / w
                        ymax = int(box.find("ymax").text) / h
                        writable.write(f"{MAPPING[ts]} "
                                       f"{(xmin + xmax) / 2:.4f} "
                                       f"{(ymin + ymax) / 2:.4f} "
                                       f"{xmax - xmin:.4f} "
                                       f"{ymax - ymin:.4f}\n")
                os.remove(fp)


def organize_structure():
    """整理目录结构，将图片和标注分别放到images和labels子目录"""
    for subset in ["train", "val", "test"]:
        subset_path = os.path.join(OUTPUT_ROOT, subset)
        if not os.path.exists(subset_path):
            continue
        
        images_path = os.path.join(subset_path, "images")
        labels_path = os.path.join(subset_path, "labels")
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)
        
        # 移动文件到相应子目录
        for f in os.listdir(subset_path):
            src = os.path.join(subset_path, f)
            if os.path.isfile(src):
                if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp'):
                    shutil.move(src, os.path.join(images_path, f))
                elif f.endswith('.txt'):
                    shutil.move(src, os.path.join(labels_path, f))


def generate_data_yaml():
    with open(os.path.join(OUTPUT_ROOT, "data.yaml"), 'w') as w:
        w.write(
            f"path: {os.path.abspath(OUTPUT_ROOT)}\n"
            "train: train\n"
            "val: val\n"
            "test: test\n"
            "\n\n"
            "names:\n" +
            "\n".join([f"  {v}: {k}" for k, v in MAPPING.items()])
        )


if __name__ == '__main__':
    handle_plain()
    post_conversion()
    organize_structure()
    generate_data_yaml()