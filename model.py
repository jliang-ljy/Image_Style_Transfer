import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights  # 导入 VGG19 和权重

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Style Transfer Model Definition
class StyleTransferModel:
    def __init__(self, content_image, style_image):
        self.content_image = content_image.to(device)
        self.style_image = style_image.to(device)
        self.generated_image = content_image.clone().to(device).requires_grad_(True)
        self.model = self.load_vgg_model()
        self.optimizer = optim.Adam([self.generated_image], lr=0.0001)

    def load_vgg_model(self):
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg

    def compute_content_loss(self, content_features, generated_features):
        return torch.mean((content_features - generated_features) ** 2)

    def compute_style_loss(self, style_grams, generated_grams):
        style_loss = 0
        for s, g in zip(style_grams, generated_grams):
            style_loss += torch.mean((s - g) ** 2)
        return style_loss

    def compute_gram_matrix(self, tensor):
        b, c, h, w = tensor.size()
        tensor = tensor.view(c, h * w)
        gram_matrix = torch.mm(tensor, tensor.t())
        return gram_matrix / (c * h * w)

    def extract_features(self, image):
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '28': 'conv5_1'}
        features = {}
        x = image
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def train(self, average_style_features, num_steps=1000):
        content_features = self.extract_features(self.content_image)['conv4_1']
        style_features = self.extract_features(self.style_image)
        style_grams = {layer: self.compute_gram_matrix(style_features[layer]) for layer in style_features}

        for step in range(num_steps):
            self.optimizer.zero_grad()
            generated_features = self.extract_features(self.generated_image)
            generated_grams = {layer: self.compute_gram_matrix(generated_features[layer]) for layer in generated_features}

            content_loss = self.compute_content_loss(content_features, generated_features['conv4_1'])
            style_loss = self.compute_style_loss(style_grams.values(), generated_grams.values())
            total_loss = content_loss + 300 * style_loss

            total_loss.backward()
            self.optimizer.step()

            if step % 50 == 0:
                print(f"Step {step}, Total Loss: {total_loss.item()}")

        return self.generated_image
