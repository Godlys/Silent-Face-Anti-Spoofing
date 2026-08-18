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
![整体架构图](https://github.com/Godlys/Silent-Face-Anti-Spoofing/blob/master/images/framework.jpg?raw=true)  

使用自研的模型剪枝方法，将MobileFaceNet的Flops从0.224G降低待了0.081G，在精度损失不大的情况下,明显提升模型的性能(降低计算量与参数量).  

| 模型架构 (Backbone) | 计算量 (FLOPs) | 参数量 (Params) | 相对 MobileFaceNet 算力降幅 |
| :--- | :---: | :---: | :---: |
| MobileFaceNet (Baseline) | 0.224 G | 0.991 M | - |
| MiniFASNetV1 | 0.081 G | 0.414 M | 63.8% ↓ |
| **MiniFASNetV2 (本项目采用)** | **0.081 G** | **0.435 M** | **63.8% ↓** |

> **说明**：在保留高频纹理判别能力的前提下，模型仅需 ~81 MFLOPs 算力开销；且在转换阶段剔除了训练阶段专用的傅里叶辅助生成分支（FTGenerator），确保推理图极致纯粹。

  工程部署优化收益 (Deployment Enhancements)
在完成向 TFLite 的计算图压制与迁移后，获得了以下端侧运行收益：
- **NHWC 内存排布优化**：对齐 ARM Cortex 架构 NEON 指令集的数据连续访问特性，消除运行时 Transpose 算子开销。
- **单运行时统一 (Single Runtime)**：彻底剔除 NDK/NCNN 运行时，与主干人脸特征提取引擎复用同一 TFLite 实例，节省内存开销并避免多框架线程竞争。
- **零拷贝数据管线**：采用 Java 直读 Native ByteBuffer 的方式直传像素流，消除了 JNI 跨层图像深拷贝带来的 GC 抖动。

## APK
### APK源码-小视科技版本
开源了适用于安卓平台的部署代码：https://github.com/minivision-ai/Silent-Face-Anti-Spoofing-APK  
### 本项目重新立项版本
* 项目地址: https://github.com/Godlys/lowFace
* 已经编译好了apk,直接下载就行,仅有适用于 armV8的版本

### Demo
<img src="https://github.com/Godlys/Silent-Face-Anti-Spoofing/blob/master/images/demo.gif?raw=true" width="300" height="400"/>  
 
### 关键指标  
| Model(input 80x80)|FLOPs|Speed| FPR | TPR |备注 |
| :------:|:-----:|:-----:| :----: | :----: | :----: |
|   APK模型 |84M| 20ms | 1e-5|97.8%| 开源|

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
![patch demo](https://github.com/Godlys/Silent-Face-Anti-Spoofing/blob/master/images/patch_demo.png?raw=true)  

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
## 更新日志
### 📦 轻量级静默活体模型迁移与边缘端优化说明

#### 1. 核心技术实施点 (Technical Engineering)

- **计算图重构与格式转换 (Graph Re-engineering & Export)**
  - 将原有的 PyTorch 动态图权重解析并固化为标准化 ONNX 中间表示（IR），最终压制编译为针对移动端/嵌入式优化的 TensorFlow Lite (`.tflite`) FlatBuffer 格式。
- **张量内存布局重排 (NCHW → NHWC Layout Optimization)**
  - 针对 ARM 架构 CPU 的 NEON 向量指令集特点，在转换阶段自动重构算子计算图，将通道优先（`NCHW`）转置为内存连续的数据优先（`NHWC`）排布，彻底消除移动端运行时（Runtime）多余的张量转置开销。
- **统一轻量级推理引擎 (Single-Runtime Architecture)**
  - 彻底剥离原方案中笨重且独立的 C++/NDK 底层推理框架与 JNI 桥接层，与主干人脸识别链路共用统一的 TFLite/LiteRT 运行时，避免在受限内存设备上并存多个深度学习引擎导致的内存碎片化与 OOM 风险。
- **像素级输入流水线校准 (Pre-processing Pipeline Alignment)**
  - **空间上下文补全**：实施 2.7 倍人脸边界框空间外延，确保高频反光与边缘失真特征完整输入。
  - **等比无畸变投影**：采用严格 1:1 正方形安全裁切与区域重采样算法，杜绝多尺度缩放产生的人脸拉伸形变。
  - **动态范围对齐**：精确匹配模型训练分布，采用原始 `[0.0, 255.0]` 浮点像素与 BGR 通道序列直接驱动卷积核，确保特征响应最大化。
- **云端无头构建流水线 (Headless CI/CD Automation)**
  - 构建全自动化的计算图转换与算子兼容性修复工作流，实现一键式自动化产物导出与校验。

---

#### 2. 适用场景与工程价值 (Applicable Scenarios & Value)

- **极低算力/内存受限边缘终端**
  - 专为低功耗 ARM 架构（如 Cortex-A53 核心集群、2GB RAM 或更低规格的 Android 工控主板、考勤门禁闸机）设计。
  - 模型仅约 2.7MB，在入门级 CPU 上单帧纯推理开销控制在几十毫秒（~70ms）级别，内存与算力开销极低。
- **单目 RGB 纯视觉静默防伪 (Passive PAD)**
  - 无需红外（IR）、结构光或 3D ToF 等特殊硬件外设，仅依赖常规单目彩色摄像头即可实现被动式、无感静默防伪，精准拦截打印照片、电子屏幕翻拍及 3D 仿生面具攻击。
- **算力级联前置拦截器 (Cascade Pre-filtering)**
  - 可作为重度人脸特征提取（1:1 / 1:N 比对）系统的前置安全网关，在毫秒级时间内过滤非法攻击请求，大幅节省主干特征网络的无效算力开销。

---

#### 3. 模型技术规格 (Technical Specifications)

- **输入张量**：`[1, 80, 80, 3]`（NHWC 排布，Float32 浮点型，BGR 色彩空间，`[0.0, 255.0]` 未归一化输入）
- **输出张量**：`[1, 3]`（Float32 Logits 分类：`2D攻击` / `真人活体` / `3D伪造`）
