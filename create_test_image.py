import cv2
import numpy as np

# 创建一个简单的测试图片
def create_test_image():
    # 创建640x640的黑色背景
    image = np.zeros((640, 640, 3), dtype=np.uint8)

    # 添加一些矩形模拟PCB上的元件
    cv2.rectangle(image, (100, 100), (200, 150), (0, 255, 0), -1)  # 绿色矩形
    cv2.rectangle(image, (300, 200), (400, 250), (255, 0, 0), -1)  # 蓝色矩形
    cv2.rectangle(image, (500, 300), (600, 350), (0, 0, 255), -1)  # 红色矩形

    # 添加一些圆形模拟焊点
    cv2.circle(image, (150, 300), 20, (255, 255, 0), -1)  # 青色圆形
    cv2.circle(image, (450, 400), 15, (255, 0, 255), -1)  # 洋红色圆形

    # 添加一些线条模拟电路
    cv2.line(image, (50, 50), (590, 50), (255, 255, 255), 3)  # 白色线条
    cv2.line(image, (50, 590), (590, 590), (255, 255, 255), 3)  # 白色线条
    cv2.line(image, (50, 50), (50, 590), (255, 255, 255), 3)  # 白色线条
    cv2.line(image, (590, 50), (590, 590), (255, 255, 255), 3)  # 白色线条

    return image

if __name__ == "__main__":
    # 确保data目录存在
    import os
    os.makedirs("data", exist_ok=True)

    # 创建测试图片
    test_image = create_test_image()

    # 保存图片
    cv2.imwrite("data/test_pcb.jpg", test_image)
    print("测试图片已保存到 data/test_pcb.jpg")