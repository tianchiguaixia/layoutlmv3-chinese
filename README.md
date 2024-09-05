## layoutlmv3 中文表单图片信息抽取
### 背景
该项目是为了使用layoutlmv3针对中文图片训练和推理。
其中主要解决三个问题：
- 1.数据标准化成可以的训练数据集格式
- 2.layoutlmv3-base-chinese 分词修改
- 2.超过512长度的文本切分和滑窗操作



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

### 模型训练
![image](https://github.com/user-attachments/assets/aac73cb8-22bb-45e3-8a07-f3f39bb1cc20)


### 模型推理效果


![1 (5)](https://github.com/user-attachments/assets/832e2990-a642-4d71-bb8e-b90d9b2bcb59)
