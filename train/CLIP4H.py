import torch
import model.HashNet as HashNet
from train.MSRVTT import MSRVTT_train_dataset, MSRVTT_val_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from utils import get_plotter
import numpy as np


class CLIP4H:
    def __init__(self):
        self.epoch = 200
        self.batch_size = 3
        self.lr = 0.01
        self.lr_decay_epoch = 150
        self.lr_decay = 0.1
        self.bit_size = 256

        self.model = HashNet.get_model(self.bit_size).to('cuda', dtype=torch.float32)
        self.weight_decay = 0.0005
        self.momentum = 0.9
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                         weight_decay=self.weight_decay)
        self.scheduler = StepLR(self.optimizer, self.lr_decay_epoch, self.lr_decay)
        self.plotter = get_plotter("CLIP4H")

    def train(self):
        train_set = MSRVTT_train_dataset()
        valid_set = MSRVTT_val_dataset()
        train_loader = DataLoader(train_set, batch_size=self.batch_size)
        for epoch in range(self.epoch):
            for image_feature, text_feature in tqdm(train_loader):  # Fv and Ft
                F_I = image_feature.to('cuda', dtype=torch.float32)
                F_T = text_feature.to('cuda', dtype=torch.float32)

                # Construct cross-modal affinity matrix
                F_I_n = F.normalize(F_I)
                F_T_n = F.normalize(F_T)

                # tensor.mm(tensor): matrix multiple
                # tensor.t(): matrix transposition
                S_IT = F_I_n.mm(F_T_n.t())
                S_TI = F_T_n.mm(F_I_n.t())

                # Set diagonal elements to 1
                # torch.diag_embed: get diagonal elements from tensor
                complete_S_IT_diagonal = torch.diag_embed(1 - S_IT.diagonal())
                complete_S_TI_diagonal = torch.diag_embed(1 - S_TI.diagonal())
                S_IT = S_IT + complete_S_IT_diagonal
                S_TI = S_TI + complete_S_TI_diagonal

                # CLIP_base
                S_C = 0.5 * S_TI + 0.5 * S_IT

                # dynamic weighting
                # use .data and Variable() to avoid in-place bug
                S_buffer = dynamic_weighting(S_C.data)
                S = Variable(S_buffer)

                # HashNet
                # H m*z m = batch_size z = bit_size
                hid_I, code_I = self.model(F_I)
                hid_T, code_T = self.model(F_T)

                H_I = F.normalize(hid_I)
                H_T = F.normalize(hid_T)

                # m*m
                HI_HI = H_I.mm(H_I.t())
                HT_HT = H_T.mm(H_T.t())
                HI_HT = H_I.mm(H_T.t())
                HT_HI = H_T.mm(H_I.t())

                # ||A||²F = ∑i∑j aij²
                # mse_loss = ∑i∑j aij² / batch_size (m)
                # each loss / m, so no difference
                intra_loss = F.mse_loss(HI_HI, S) + F.mse_loss(HT_HT, S)
                inter_loss = F.mse_loss(HI_HT, S) + F.mse_loss(HT_HI, S)
                consistency_loss = F.mse_loss(H_I, H_T)

                loss = 0.1 * intra_loss + 1 * inter_loss + 2 * consistency_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            valid(epoch, valid_set, self.model, self.batch_size)
            self.scheduler.step()
            self.plotter.next_epoch()


def dynamic_weighting(S):
    # mean min max values of Sc in each batch
    S_mean = torch.mean(S)
    S_min = torch.min(S)
    S_max = torch.max(S)
    # torch.exp(x) = e^x
    # S[S < S_mean] includes all elements that < S_mean
    # = Si,j
    W_low = torch.exp(-0.5 * (S_mean - S[S <= S_mean]) / (S_mean - S_min) - 0.5)
    S[S <= S_mean] = W_low * S[S <= S_mean]

    W_high = torch.exp(0.5 * (S[S > S_mean] - S_mean) / (S_max - S_mean) - 0.5)
    S[S > S_mean] = W_high * S[S > S_mean]

    S[S > 1.0] = 1.0
    S[S < -1.0] = -1.0
    return S


def valid(epoch, valid_set, model, batch_size):
    #  1000 * bit_size
    query_B_img, query_B_txt = get_valid_codes(valid_set, model, batch_size)
    retrieve_B_img, retrieve_B_txt = get_valid_codes(valid_set, model, batch_size)
    gt_path = 'dataset/MSRVTT/ground_truth.txt'
    gt = np.loadtxt(gt_path, dtype=int).reshape(1000, 1000)

    return 1


def get_valid_codes(valid_set, HashNet_model, batch_size):
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    img_buffer = txt_buffer = torch.empty(len(valid_set), HashNet_model.bit_size, dtype=torch.float32)
    img_buffer, txt_buffer = to_cuda(img_buffer, txt_buffer)
    index = 0
    for image_feature, text_feature in tqdm(valid_loader):  # Fv and Ft
        F_I = image_feature.to('cuda', dtype=torch.float32)
        F_T = text_feature.to('cuda', dtype=torch.float32)
        hid_I, code_I = HashNet_model(F_I)
        hid_T, code_T = HashNet_model(F_T)
        img_buffer[index, :] = code_I.data
        txt_buffer[index, :] = code_T.data
        index = index + 1
    return img_buffer, txt_buffer


def to_cuda(*args):
    """
    chagne all tensor from cpu tensor to cuda tensor
    :param args: tensors or models
    :return: the tensors or models in cuda by input order
    """
    cuda_args = []
    for arg in args:
        cuda_args.append(arg.cuda())
    return cuda_args


def train():
    trainer = CLIP4H()
    trainer.train()