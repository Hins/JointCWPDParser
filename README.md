# 基于jptdp的句法分析模型推理工具

## 目录

+ <a href="#1">功能介绍</a>
+ <a href="#2">上手指南</a>
  + <a href="#3">开发前的配置要求</a>
  + <a href="#4">安装步骤</a>
+ <a href="#5">文件目录说明</a>

## <span name="1">功能介绍</span>

​		基于jptdp的句法分析模型推理工具，针对中文句法分析输出结果。输入的格式为 .txt 输出格式为 .json。

##<span name="2">上手指南 </span>

### <span name="3">开发前的配置要求</span>

arm服务器
psutil
pathlib
jieba
torch
numpy
argparse

### <span name="4">安装步骤</span>

pip install -r requirements.txt

## <span name="5">文件目录说明</span>

code
├── README.md ---> 工具说明
├── Dockerfile ---> docker镜像工具
├── /TuneBertJointCWPD/ ---> 模型训练文件夹
│ ├── inference.py ---> 推理工具
│ ├── monitoring.py ---> 监控工具
│ ├── /conf/ ---> 配置工具
│ ├── /datautil/ ---> util工具
│ ├── /log/ ---> 日志工具
│ ├── /vocab/ ---> 字典工具
│ ├── /modules/ ---> 模型工具