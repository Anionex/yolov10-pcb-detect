### 简介:
第十九届“挑战杯”揭榜挂帅·华为赛道（高校赛道）Rank35/320方案.使用在coco数据集上经过预训练的yolov10s模型进行迁移学习.
### 分类:
- 深度学习/计算机视觉
### TODO
- [✔] fix nms and red pcb defection
- [✔] change conf and slide method


### 项目结构解释：
- data: 临时测试数据存放文件夹，用来测试costomize_service的可用性，costomize_service被test_service调用<
- tmp_output: test_service的输出
- service_app.py是一个gradio应用，方便标注可视化和预测可视化，还加了一些图片处理方法
- app-origin.py为官方提供的gradio应用，用作基本的预测，貌似支持视频。
- utils/存放数据处理工具
