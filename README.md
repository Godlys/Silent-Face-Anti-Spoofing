# 静默活体检测 (Silent-Face-Anti-Spoofing)   
该项目为[小视科技]的静默活体检测项目的复制版本.
## 更新  
**2026-08-14:** 写了使用GitHub Action 编译转换，将原有的 PyTorch 动态图权重解析并固化为标准化 ONNX 中间表示（IR），最终压制编译为针对移动端/嵌入式优化的 TensorFlow Lite (`.tflite`) FlatBuffer 格式。
**2020-07-30:** 开源caffe模型，分享工业级静默活体检测算法技术解析直播视频以及相关文件。

## 简介
在本工程中我们开源了活体模型训练架构，数据预处理方法，模型训练和测试脚本以及开源的APK供大家测试使用。  
本项目是对于[https://github.com/Godlys/Silent-Face-Anti-Spoofing] 仓库做了一套将 PyTorch 编译为 TensorFlow Lite 的权重模型

活体检测技术主要是判别机器前出现的人脸是真实还是伪造的，其中借助其他媒介呈现的人脸都可以定义为虚假的人脸，包括打印的纸质照片、电子产品的显示屏幕、硅胶面具、立体的3D人像等。目前主流的活体解决方案分为配合式活体检测和非配合式活体检测（静默活体检测）。配合式活体检测需要用户根据提示完成指定的动作，然后再进行活体校验，静默活体则在用户无感的情况下直接进行活体校验。  
 
因傅里叶频谱图一定程度上能够反应真假脸在频域的差异,因此我们采用了一种基于傅里叶频谱图辅助监督的静默活体检测方法, 模型架构由分类主分支和傅里叶频谱图辅助监督分支构成，整体架构如下图所示：  
![整体架构图]([https://github.com/Godlys/Silent-Face-Anti-Spoofing/blob/master/images/framework.jpg?raw=true)  

使用自研的模型剪枝方法，将MobileFaceNet的Flops从0.224G降低待了0.081G，在精度损失不大的情况下,明显提升模型的性能(降低计算量与参数量).  

|Model|FLOPs|Params|
| :------:|:-----:|:-----:| 
|MobileFaceNet|0.224G|0.991M|
|MiniFASNetV1|0.081G|0.414M|
|MiniFASNetV2|0.081G|0.435M|

## APK
### APK源码-小视科技版本
开源了适用于安卓平台的部署代码：https://github.com/minivision-ai/Silent-Face-Anti-Spoofing-APK  
### 本项目重新立项版本
* 项目地址: https://github.com/Godlys/lowFace
* 已经编译好了apk,直接下载就行,仅有适用于 armV8的版本

### Demo
<img src="[https://github.com/Godlys/Silent-Face-Anti-Spoofing/blob/master/images/demo.gif?raw=true" width="300" height="400"/>  
 
### 关键指标  
| Model(input 80x80)|FLOPs|Speed| FPR | TPR |备注 |
| :------:|:-----:|:-----:| :----: | :----: | :----: |
|   APK模型 |84M| 20ms | 1e-5|97.8%| 开源|
| 高精度模型 |162M| 40ms| 1e-5 |99.7%| 未开源 |

### 测试方法  
- 显示信息:速度(ms), 置信度(0~1)以及活体检测结果(真脸or假脸)
- 点击右上角图标可设置阈值,如果置信度大于阈值,为真脸,否则为假脸  

### 测试须知 
- 所有测试图片必须通过摄像头采集得到，否则不符合正常场景使用规范，算法效果也无法保证。
- 因为RGB静默活体对摄像头型号和使用场景鲁棒性受限，所以实际使用体验会有一定差异。
- 测试时，应保证有完整的人脸出现在视图中，并且人脸旋转角与竖直方向小于30度（符合正常刷脸场景），否则影响体验。　　

**已测试型号**

|型号|麒麟990 5G|麒麟990 |骁龙845 |麒麟810 |RK3288 |
| :------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|速度/ms|19|23|24|25|90|

## 工程
### 安装依赖库  
```
pip install -r requirements.txt
```
### Clone
```
git clone [https://github.com/Godlys/Silent-Face-Anti-Spoofing]
cd Silent-Face-Anti-Spoofing
```  
### 数据预处理
1.将训练集分为3类,将相同类别的图片放入一个文件夹;  
2.因采用多尺度模型融合的方法,分别用原图和不同的patch训练模型,所以将数据分为原图和基于原图的patch;  
- 原图(org_1_height**x**width),直接将原图resize到固定尺寸(width, height),如图1所示;  
- 基于原图的patch(scale_height**x**width),采用人脸检测器人脸,获取人脸框,按照一定比例(scale)对人脸框进行扩边，为了保证模型的输入尺寸的一致性，将人脸框区域resize到固定尺寸(width, height),图2-4分别显示了scale为1,2.7和4的patch样例;  
![patch demo]([https://github.com/Godlys/Silent-Face-Anti-Spoofing/blob/master/images/patch_demo.png)  

3.采用傅里叶频谱图作为辅助监督,训练集图片在线生成对应的傅里叶频谱图.  
**数据集的目录结构如下所示**
```
├── datasets
    └── RGB_Images
        ├── org_1_80x60
            ├── 0
		├── aaa.png
		├── bbb.png
		└── ...
            ├── 1
		├── ddd.png
		├── eee.png
		└── ...
            └── 2
		├── ggg.png
		├── hhh.png
		└── ...
        ├── 1_80x80
        └── ...
```  
### 训练
```
python train.py --device_ids 0  --patch_info your_patch
```  
### 测试
 ./resources/anti_spoof_models 活体检测的融合模型  
 ./resources/detection_model 检测器模型  
 ./images/sample 测试图片  
 ```
 python test.py --image_name your_image_name
 ```      
