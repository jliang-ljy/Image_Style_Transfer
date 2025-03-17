# utils.py
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path, max_size=256):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),  # 调整到固定大小
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)  # 增加 batch 维度
    return image


def load_images_from_folder(folder_path, max_size=400):
    images = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            images.append(load_image(image_path, max_size=max_size))
    return images

def save_image(tensor, path):
    image = tensor.clone().detach().squeeze(0)
    image = transforms.ToPILImage()(image.cpu())
    image.save(path)

def imshow(tensor, title=None):
    image = tensor.clone().detach().squeeze(0)
    image = transforms.ToPILImage()(image.cpu())
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()