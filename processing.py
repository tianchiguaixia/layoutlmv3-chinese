import os
import json
import uuid
from PIL import Image
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Array2D, Array3D, Value
from transformers import LayoutXLMTokenizer, LayoutLMv3ImageProcessor, LayoutLMv3Processor

# 创建输出目录
OUTPUT_PATH = "./content/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 读取 JSON 数据
with open("zh.train.json") as f:
    file_data = json.load(f)

# 标签映射
label2id = {"O": 0, "B-header": 1, "I-header": 2, "B-question": 3, "I-question": 4, "B-answer": 5, "I-answer": 6}
id2label = {v: k for k, v in label2id.items()}

# 标准化 bbox 坐标
def normalize_box(bbox, width, height):
    return [int((bbox[0] / width) * 1000), int((bbox[1] / height) * 1000),
            int((bbox[2] / width) * 1000), int((bbox[3] / height) * 1000)]

# 滑动窗口函数
def sliding_window(data, max_length, stride):
    return [data[i:i + max_length] for i in range(0, len(data), stride)]

# 滑窗参数
max_length, stride = 400, 30

# 初始化存储列表
words_windows, bboxes_windows, ner_tags_windows, image_windows, ids = [], [], [], [], []

# 处理数据
for doc in file_data["documents"]:
    image = Image.open(f"zh/{doc['id']}.jpg")
    width, height = image.size
    tokens, bboxes, ner_tags = [], [], []

    # 遍历每个文档的元素
    for element in doc["document"]:
        label = element["label"]
        label_map = {"header": (1, 2), "question": (3, 4), "answer": (5, 6), "other": (0, 0)}
        b_tag, i_tag = label_map.get(label, (0, 0))

        # 提取每个单词及其对应的 bbox 和标签
        for idx, word in enumerate(element["words"]):
            tokens.append(word["text"])
            bboxes.append(normalize_box(word["box"], width, height))
            ner_tags.append(b_tag if idx == 0 else i_tag)

    # 滑窗处理
    word_chunks = sliding_window(tokens, max_length, stride)
    bbox_chunks = sliding_window(bboxes, max_length, stride)
    ner_tag_chunks = sliding_window(ner_tags, max_length, stride)

    # 将窗口添加到最终列表
    words_windows.extend(word_chunks)
    bboxes_windows.extend(bbox_chunks)
    ner_tags_windows.extend(ner_tag_chunks)
    image_windows.extend([image] * len(word_chunks))
    ids.extend([str(uuid.uuid4())] * len(word_chunks))

# 构建数据集
data = {
    'id': ids,
    'tokens': words_windows,
    'bboxes': bboxes_windows,
    'ner_tags': ner_tags_windows,
    'image': image_windows
}

# 创建 NER 标签特征
ner_feature = Sequence(ClassLabel(names=[id2label[i] for i in range(len(id2label))]))

# 构建 Dataset 并为 ner_tags 指定特征
train_dataset = Dataset.from_dict(data).cast_column('ner_tags', ner_feature)
test_dataset = Dataset.from_dict(data).cast_column('ner_tags', ner_feature)

# 构建 DatasetDict
dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

# 输出 labels
labels = dataset['train'].features['ner_tags'].feature.names

# 加载 Tokenizer 和 ImageProcessor
tokenizer = LayoutXLMTokenizer.from_pretrained("/home/cg.peng/layoutlm-model/layoutlmv3-base-chinese")
image_processor = LayoutLMv3ImageProcessor.from_pretrained("/home/cg.peng/layoutlm-model/layoutlmv3-base-chinese", apply_ocr=False)

# 创建 Processor
processor = LayoutLMv3Processor(tokenizer=tokenizer, image_processor=image_processor, apply_ocr=False)

# 定义 prepare_examples 函数
def prepare_examples(examples):
    images = examples["image"]
    words = examples["tokens"]
    boxes = examples["bboxes"]
    word_labels = examples["ner_tags"]
    
    return processor(images, words, boxes=boxes, word_labels=word_labels, truncation=True, padding="max_length")

# 特征定义
features = Features({
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
    'input_ids': Sequence(Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(ClassLabel(names=labels)),
})

# 处理训练集和验证集
train_dataset = dataset["train"].map(prepare_examples, batched=True, remove_columns=dataset["train"].column_names, features=features)
eval_dataset = dataset["test"].map(prepare_examples, batched=True, remove_columns=dataset["test"].column_names, features=features)

# 将数据集格式设为 torch
train_dataset.set_format("torch")
eval_dataset.set_format("torch")

# 保存数据集
train_dataset.save_to_disk(f'{OUTPUT_PATH}train_split')
eval_dataset.save_to_disk(f'{OUTPUT_PATH}eval_split')
dataset.save_to_disk(f'{OUTPUT_PATH}raw_data')
