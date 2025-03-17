# Imports and Environment Setup
import torch
import os
from utils import load_image, save_image, imshow, load_images_from_folder
from model import StyleTransferModel

torch.backends.cudnn.enabled = False  # 临时禁用 cuDNN

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load content image
content_image = load_image("data\content_1.jpg").to(device)

# Load all style images
style_images = load_images_from_folder("data/trainC")
style_images = [img.to(device) for img in style_images]

# Compute average style features
#At this stage, the model is not trained, only needs to be initialized to extract features
#VGG19 is a pre-trained model used to extract features from the style images
model = StyleTransferModel(content_image, style_images[0])  # Initialize once to use 

# 分批处理风格图片
batch_size = 100  # 每次处理 10 张图片
style_features_list = []

print(f"Processing {len(style_images)} style images in batches of {batch_size}...")
for i in range(0, len(style_images), batch_size):
    batch = style_images[i:i + batch_size]  # 提取当前批次
    for style_image in batch:
        style_features = model.extract_features(style_image)  # 提取特征
        style_features_list.append(style_features)  # 保存特征
    torch.cuda.empty_cache()  # 清理显存，避免内存碎片化
    print(f"Processed batch {i // batch_size + 1}/{(len(style_images) + batch_size - 1) // batch_size}")

print("Successfully extracted features for one style image.")
# Compute the average style features
average_style_features = {key: sum(features[key] for features in style_features_list) / len(style_features_list)
                          for key in style_features_list[0].keys()}

# Train the model with the average style features
model = StyleTransferModel(content_image, style_images[0])  # Reuse model
#The model is trained with the average style features
output_image = model.train(average_style_features, num_steps=1000)

# Save the result
save_path = "D:/pj_style_transfer/generated_images/final_result.jpg"
save_image(output_image, save_path)
print(f"The generated image has been saved to {save_path}")
