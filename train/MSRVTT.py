from PIL import Image
from model.pretrained_clip import clip
import torch
import torch.utils.data as data


class MSRVTT_val_dataset(data.Dataset):
    def __init__(self):
        self.total_len = 1000
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pretrained_clip_path = 'model/pretrained_clip/ViT-B-32.pt'
        self.clip, self.preprocess = clip.load(self.pretrained_clip_path, device=self.device)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        num = 12
        image_features = []
        for i in range(num):
            image = self.preprocess(Image.open(f"dataset/MSRVTT/frame/video{9000 + index}frame{i}.jpg")).unsqueeze(0).to(self.device)
            image_feature = self.clip.encode_image(image)
            image_features.append(image_feature)
        # 12*1*512
        image_batch = torch.stack(image_features)
        # 1*512
        averaged_image_feature = torch.mean(image_batch, dim=0)

        text_features = []
        with open(f'dataset/MSRVTT/caption/video{9000 + index}.txt', 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                text = clip.tokenize(line).to(self.device)
                text_feature = self.clip.encode_text(text)
                text_features.append(text_feature)
        # 20*1*512
        text_batch = torch.stack(text_features)
        # 1*512
        averaged_text_feature = torch.mean(text_batch, dim=0)
        return averaged_image_feature.squeeze(), averaged_text_feature.squeeze()


class MSRVTT_train_dataset(data.Dataset):
    def __init__(self):
        self.total_len = 9000
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pretrained_clip_path = 'model/pretrained_clip/ViT-B-32.pt'
        self.clip, self.preprocess = clip.load(self.pretrained_clip_path, device=self.device)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        num = 12
        image_features = []
        for i in range(num):
            image = self.preprocess(Image.open(f"dataset/MSRVTT/frame/video{index}frame{i}.jpg")).unsqueeze(0).to(self.device)
            image_feature = self.clip.encode_image(image)
            image_features.append(image_feature)
        # 12*1*512
        image_batch = torch.stack(image_features)
        # 1*512
        averaged_image_feature = torch.mean(image_batch, dim=0)

        text_features = []
        with open(f'dataset/MSRVTT/caption/video{index}.txt', 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                text = clip.tokenize(line).to(self.device)
                text_feature = self.clip.encode_text(text)
                text_features.append(text_feature)
        # 20*1*512
        text_batch = torch.stack(text_features)
        # 1*512
        averaged_text_feature = torch.mean(text_batch, dim=0)
        return averaged_image_feature.squeeze(), averaged_text_feature.squeeze()
