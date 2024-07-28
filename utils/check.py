import os

#得到当前目录下images/下的所有jpg文件
def get_images():
    # 支持的图片格式
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    # 获取文件列表
    file_list = [ f for f in os.listdir(images_directory) if f.lower().endswith(supported_formats)]
    total_files = len(file_list)

    if total_files == 0:
        print("没有找到任何支持的图片格式。")
        return

    print(f"found {total_files} images")
    return file_list
    

def get_labels():
    # 支持的图片格式
    supported_formats = (".txt", )

    # 获取文件列表
    file_list = [ f for f in os.listdir(labels_directory) if f.lower().endswith(supported_formats)]
    total_files = len(file_list)

    if total_files == 0:
        print("没有找到任何支持的标注格式。")
        return

    print(f"found {total_files} annos")
    return file_list

# 检查每个jpg文件是否有一个同名的txt文件在当前目录下的labels/中
def check_labels(images):
    ret = True
    cnt = 0
    for image in images:
        label = image.replace('.jpg', '.txt')
        if not os.path.exists(os.path.join(labels_directory, label)):
            print(f"no label file for {image}",end="")
            os.remove(os.path.join(images_directory, image))
            ret = False
            cnt +=1
        else:
            # print(f"found label file for {image}")
            # delete the label in the list[] labels
            labels.remove(label)
            
    return ret, cnt

current_directory = os.path.dirname(os.path.abspath(__file__))
images_directory = os.path.join(current_directory, "images")
labels_directory = os.path.join(current_directory, "labels")

    
images = get_images()
labels = get_labels()
res, cnt = check_labels(images)
if res:
    print("all images have labels")
else:
    print(f"{cnt} images do not have labels!!!!")
    
for i in range(len(labels)):
    # print(f"extra label file: {labels[i]}  ",end="")
    os.remove(os.path.join(labels_directory, labels[i]))