# 导入必要的库
from datasets import load_metric
from transformers import TrainingArguments, Trainer
from transformers import LayoutLMv3ForTokenClassification, AutoProcessor,LayoutXLMTokenizer,LayoutLMv3ImageProcessor,LayoutLMv3Processor
from transformers.data.data_collator import default_data_collator
import torch
from datasets import load_from_disk
import numpy as np

# 加载已经预处理好的训练和验证数据集
train_dataset = load_from_disk(f'./content/train_split')
eval_dataset = load_from_disk(f'./content/eval_split')

# 获取标签列表及其映射关系
label_list = train_dataset.features["labels"].feature.names  # 获取标签名称列表
num_labels = len(label_list)  # 标签数量
label2id, id2label = dict(), dict()  # 创建标签到ID的映射和ID到标签的映射
for i, label in enumerate(label_list):
    label2id[label] = i  # 标签对应的ID
    id2label[i] = label  # ID对应的标签
    
# 加载性能评估的指标
metric = load_metric("seqeval")

# 是否返回实体级别的评价指标
return_entity_level_metrics = False

# 定义模型评估时的计算指标函数
def compute_metrics(p):
    # 将模型预测结果转化为标签ID
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)  # 取每个位置上概率最大的标签ID

    # 提取真实的预测标签和真实的标签，并过滤掉标签为 -100 的填充部分
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # 计算各类指标，如精度、召回率、F1值等
    results = metric.compute(predictions=true_predictions, references=true_labels, zero_division='0')

    # 如果需要返回实体级别的评价指标
    if return_entity_level_metrics:
        final_results = {}
        # 解析并展开嵌套字典
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        # 返回整体指标
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

# 加载训练好的 LayoutLMv3 模型，用于Token分类
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "/home/cg.peng/layoutlm-model/layoutlmv3-base-chinese",
    id2label=id2label,  # 将模型的输出与标签ID映射到对应的标签名称
    label2id=label2id   # 将标签名称映射回ID
)

# 加载 XLMRobertaTokenizer，用于将文本转化为模型输入
tokenizer = LayoutXLMTokenizer.from_pretrained(
    "/home/cg.peng/layoutlm-model/layoutlmv3-base-chinese"
)

# 加载 LayoutLMv3ImageProcessor，用于处理图像输入
image_processor = LayoutLMv3ImageProcessor.from_pretrained(
    "/home/cg.peng/layoutlm-model/layoutlmv3-base-chinese", apply_ocr=False  # 不使用 OCR 功能
)

# 创建 LayoutLMv3Processor，用于处理图像和文本的组合输入
processor = LayoutLMv3Processor(tokenizer=tokenizer, image_processor=image_processor, apply_ocr=False)

# 设置训练参数
NUM_TRAIN_EPOCHS = 10  # 训练的轮数
PER_DEVICE_TRAIN_BATCH_SIZE = 1  # 每个设备上训练时的批次大小
PER_DEVICE_EVAL_BATCH_SIZE = 1  # 每个设备上评估时的批次大小
LEARNING_RATE = 4e-5  # 学习率

# 定义训练参数
training_args = TrainingArguments(
    output_dir="test",  # 输出目录
    num_train_epochs=NUM_TRAIN_EPOCHS,  # 训练的轮数
    logging_strategy="epoch",  # 每轮日志记录一次
    save_total_limit=1,  # 保存的模型文件数量限制为1个
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,  # 训练批次大小
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,  # 评估批次大小
    learning_rate=LEARNING_RATE,  # 设置学习率
    evaluation_strategy="epoch",  # 每个epoch进行一次评估
    save_strategy="epoch",  # 每个epoch保存一次模型
    load_best_model_at_end=True,  # 在训练结束时加载最好的模型
    metric_for_best_model="f1"  # 以F1值作为选择最优模型的标准
)

# 使用 Trainer API 来训练模型
trainer = Trainer(
    model=model,  # 要训练的模型
    args=training_args,  # 训练的参数
    train_dataset=train_dataset,  # 训练数据集
    eval_dataset=train_dataset,  # 验证数据集
    tokenizer=processor,  # 用于处理输入数据的tokenizer和processor
    data_collator=default_data_collator,  # 默认的数据整理器，用于将数据打包为批次
    compute_metrics=compute_metrics,  # 计算评估指标的函数
)

# 开始训练
trainer.train()

# 评估模型
trainer.evaluate()

