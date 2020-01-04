more info pls refer [TianChi](https://tianchi.aliyun.com/forum/postDetail?postId=87366)

## 代码说明

- 模型：
使用的都是TensorFlow预训练好的模型权重，可以通过下面两个链接下载
    - [TensorFlow 对抗权重](https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models)
    - [TensorFlow Slim权重](https://github.com/tensorflow/models/tree/master/research/slim)
- 代码
     - 01_code：该文件夹存放攻击相关代码
        - `run_attack.sh`：通过脚本方式配置参数 & 调用攻击代码
        - `attack_img_50_32_6m_2.py`：attack代码
    - 05_acc: 各个模型的评测代码
    
    - `run_acc.sh`：评估原始图像识别情况，以及评估攻击效果，
        - `acc_attack.sh`：`run_acc.sh`中调用的脚本
    - `check_pixs_attack.py`：检查攻击后的图片是否像素值超过32




