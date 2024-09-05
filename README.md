## layoutlmv3 中文表单图片信息抽取
### 背景
该项目是为了使用layoutlmv3针对中文图片训练和推理。
其中主要解决三个问题：
1.数据标准化成科训练数据集
2.layoutlmv3-base-chinese 分词修改
2.超过512长度的文本切分和滑窗操作



### 数据来源
https://github.com/doc-analysis/XFUND/releases/tag/v1.0
![image](https://github.com/user-attachments/assets/c20a7fe8-fa19-43bd-851e-2ab780380c9e)

### 图片样例
![image](https://github.com/tianchiguaixia/layoutlmv3-chinese/blob/main/img/zh_train_0%20(1).jpg)

### 代码结构

```
├── processing.py     # 数据处理
├── training.py       # 模型训练
├── inference.py       #模型推理
```


