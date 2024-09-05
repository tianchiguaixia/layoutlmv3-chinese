## layoutlmv3 中文表单图片信息抽取
### 背景
该项目是为了使用layoutlmv3针对中文图片训练和推理。
其中主要解决三个问题：
1.数据标准化成科训练数据集
2.layoutlmv3-base-chinese 分词修改
2.超过512长度的文本切分和滑窗操作



### 数据形式
![](img/1.jpeg)

### 代码结构

```
├── processing.py     # 数据处理
├── training.py       # 模型训练
├── inference.py       #模型推理
```


