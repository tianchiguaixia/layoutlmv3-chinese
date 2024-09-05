from PIL import ImageDraw, ImageFont
import torch


from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("/home/cg.peng/layoutlm-model/训练/test/checkpoint-23180")


example = dataset["train"][0]




# 加载 XLMRobertaTokenizer
tokenizer = LayoutXLMTokenizer.from_pretrained(
    "/home/cg.peng/layoutlm-model/layoutlmv3-base-chinese"
)

# 加载 LayoutLMv3ImageProcessor 用于处理图像
image_processor = LayoutLMv3ImageProcessor.from_pretrained(
    "/home/cg.peng/layoutlm-model/layoutlmv3-base-chinese", apply_ocr=False
)

# 创建 LayoutLMv3Processor
processor = LayoutLMv3Processor(tokenizer=tokenizer, image_processor=image_processor, apply_ocr=False)



# 定义帮助函数
def process_chunk(image, words, boxes, word_labels, processor, model):
    encoding = processor(
        image,
        words,
        boxes=boxes,
        word_labels=word_labels,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()
    labels = encoding.labels.squeeze().tolist()

    return predictions, labels, encoding.bbox.squeeze().tolist()

def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

def iob_to_label(label):
    label = label[2]
    if not label:
        return 'other'
    return label

label2color = {'q':'blue', 'a':'green', 'h':'orange', 'other':'violet'}

def draw_predictions_on_image(image, true_predictions, true_boxes):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(box, outline=label2color.get(predicted_label, 'violet'))
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color.get(predicted_label, 'violet'), font=font)

# 查找id为001d97f3-e04f-4230-90b5-f5a5467c812c的所有样本
target_id = dataset["train"]["id"][0]
target_examples = []

for dataset_split in ['train', 'test']:
    for example in dataset[dataset_split]:
        if example['id'] == target_id:
            target_examples.append(example)

# 处理和绘制
if target_examples:
    all_predictions = []
    all_boxes = []

    # 假设所有样本共享相同的图像
    image = target_examples[0]["image"]
    width, height = image.size

    for example in target_examples:
        words = example["tokens"]
        boxes = example["bboxes"]
        word_labels = example["ner_tags"]

        predictions, labels, token_boxes = process_chunk(image, words, boxes, word_labels, processor, model)

        all_predictions.extend([model.config.id2label[pred] for pred, label in zip(predictions, labels) if label != -100])
        all_boxes.extend([unnormalize_box(box, width, height) for box, label in zip(token_boxes, labels) if label != -100])

    # 将所有预测绘制到图像上
    draw_predictions_on_image(image, all_predictions, all_boxes)

    # 显示或保存图像
    #image.show()  # 或使用 image.save(f"output_image_{target_id}.png") 保存图像
else:
    print(f"No examples found for id {target_id}")

