import os
import random
import re
import shutil
from functools import cache

DATASET_OUTPUT = "./deep_pcb_dataset"  # 数据集位置
SRC = "./DeepPCB-master/PCBData"

PIC_HEIGHT = PIC_WIDTH = 640

TRAIN = 0
TEST = 1
VAL = 2
SUBSET_MAPPING = {
    TRAIN: "train",
    TEST: "test",
    VAL: "val"
}

CLAZZ_MAPPING = {
    0: "background",
    1: "broken",  # 断路
    2: "short",  # 短路
    3: "gap",  # 缺口
    4: "burr",  # 毛刺
    5: "copper",  # 露铜
    6: "hole",  # 针孔
}

VAL_RATIO = 0.2  # 评估集比例
TEST_RATIO = 0.2  # 测试集比例

cls_transfer = {
    "0": -1,
    "1": 4,
    "2": 3,
    "3": 0,
    "4": 1,
    "5": 5,
    "6": -1,
}
@cache
def get_subset_path(data_type: int, *to_join: str, is_file=False) -> str:
    """
    获取子数据集下的路径, 此方法保证文件夹的存在性.
    Args:
        data_type: 子数据集类型, 见 SUBSET_MAPPING.
        is_file: to_join 最后一项是不是表示文件, 如果表示文件, 那么 to_join 最后一项不会被当成目录创建.
        *to_join: 要附加的子目录.
    """
    rst = os.path.join(DATASET_OUTPUT, SUBSET_MAPPING[data_type], *to_join)
    os.makedirs(os.path.dirname(rst) if is_file else rst, exist_ok=True)
    return rst


def selector(test_ratio=TEST_RATIO, val_ratio=VAL_RATIO):
    """
    用来挑选单张图片数据所属的子数据集.

    Yields:
        TRAIN 表示应该是训练集.
        TEST 表示应该是测试集.
        VAL 表示应该是评估集.
    """
    test_val_split = 1 - val_ratio
    train_ratio = test_val_split - test_ratio
    while True:
        p = random.random()
        if p < train_ratio:
            yield TRAIN
        elif p < test_val_split:
            yield TEST
        else:
            yield VAL


Selector = selector()


def convert_notation_file(filepath: str, data_type: int):
    """
    把标注文件转化成 YOLO 格式的标注文件, 并输出到 DATASET_OUTPUT 对应目录,
    此方法对 filepath 所代表的文件不产生影响.
    Args:
        filepath: 标注文件的路径.
        data_type: 所属数据集, 见 selector.
    """
    with open(filepath, 'r') as r:
        with open(get_subset_path(data_type, "labels", os.path.basename(filepath), is_file=True),
                  'w') as w:
            for line in r:
                ltx, lty, rbx, rby, clz = [
                    int(i) for i in re.match(r"(\d+) (\d+) (\d+) (\d+) (\d+)", line).groups()
                ]
                ltx = ltx / PIC_WIDTH
                lty = lty / PIC_HEIGHT
                rbx = rbx / PIC_WIDTH
                rby = rby / PIC_HEIGHT
                if cls_transfer[str(clz)] != -1:
                    w.write(f"{cls_transfer[str(clz)]} {(ltx + rbx)/2:.4f} {(lty + rby)/2:.4f} {rbx - ltx:.4f} {rby - lty:.4f}\n")


def transfer_pic_file(pic_path: str, data_type: int):
    """
    把一个图片转移到 DATASET_OUTPUT 内的相应位置并去除名称中的 _test.
    此方法对 pic_path 表示的文件不产生影响.
    Args:
        pic_path: 要转移的图片文件.
        data_type: 所属数据集, 见 selector.
    """
    pre, suf = re.match(r"(\d+)_test(.+)", os.path.basename(pic_path)).groups()
    shutil.copyfile(pic_path, get_subset_path(
        data_type,
        "images",
        pre + suf,
        is_file=True,
    ))


def generate_data_yaml():
    pass
    # """
    # 在 DATASET_OUTPUT 下生成数据集标注文件.
    # """
    # with open(os.path.join(DATASET_OUTPUT, "data.yaml"), 'w') as w:
    #     w.write(
    #         f"path: {os.path.abspath(DATASET_OUTPUT)}\n"
    #         "train: train\n"
    #         "val: val\n"
    #         "test: test\n"
    #         "\n\n"
    #         "names:\n" +
    #         "\n".join((f"  {k}: {v}" for (k, v) in CLAZZ_MAPPING.items()))
    #     )


def handle_group(group_dir: str) -> None:
    """
    把一个 group\\d{5} 文件夹下的数据提取到目标 YOLO 数据集.
    """
    group_num = os.path.basename(group_dir)[5:]
    pics_dir = os.path.join(group_dir, group_num)
    notation_dir = os.path.join(group_dir, group_num + "_not")
    for pic_name in os.listdir(pics_dir):
        matched = re.match(r"(\d+)_test.jpg", pic_name)
        if matched:
            data_type = next(Selector)
            pic_num = matched.group(1)
            pic_path = os.path.join(pics_dir, pic_name)
            notation_path = os.path.join(notation_dir, pic_num + ".txt")
            convert_notation_file(notation_path, data_type)
            transfer_pic_file(pic_path, data_type)


def main():
    for group_dir in os.listdir(SRC):
        if "group" in group_dir:
            handle_group(os.path.join(SRC, group_dir))
    # generate_data_yaml()


if __name__ == '__main__':
    main()
