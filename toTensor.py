from PIL import Image
from model.pretrained_clip import clip

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
image_tensors = {}
text_tensors = {}
ten = torch.load('dataset/MSRVTT/image_tensors.pt')
for index in range(2):
    num = 12

    image_features = []
    for i in range(num):
        image = preprocess(Image.open(f"dataset/MSRVTT/frame/video{index}frame{i}.jpg")).unsqueeze(0).to(
            device)
        image_feature = model.encode_image(image)
        image_features.append(image_feature)
    # 12*1*512
    image_batch = torch.stack(image_features)
    # 1*512
    averaged_image_feature = torch.mean(image_batch, dim=0)
    image_tensors[f'{index}'] = averaged_image_feature

    text_features = []
    with open(f'dataset/MSRVTT/caption/video{index}.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            text = clip.tokenize(line).to(device)
            text_feature = model.encode_text(text)
            text_features.append(text_feature)
            # 20*1*512
        text_batch = torch.stack(text_features)
        # 1*512
        averaged_text_feature = torch.mean(text_batch, dim=0)
        text_tensors[f'{index}'] = averaged_text_feature
'''torch.save(text_tensors, 'dataset/MSRVTT/text_tensors.pt')
torch.save(image_tensors, 'dataset/MSRVTT/image_tensors.pt')'''


