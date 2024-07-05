from datetime import datetime
import torch
import model.HashNet as HashNet
from train.MSRVTT import MSRVTT_train_dataset, MSRVTT_val_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np


class CLIP4H:
    def __init__(self):
        self.epoch = 200
        self.train_set = MSRVTT_train_dataset()
        self.valid_set = MSRVTT_val_dataset()
        self.batch_size = 16
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

        self.max_r1_i2t = 0
        self.max_r1_t2i = 0
        self.max_r5_i2t = 0
        self.max_r5_t2i = 0
        self.max_r10_i2t = 0
        self.max_r10_t2i = 0
        # MdR: median rank of GT videos
        self.min_MdR_i2t = len(self.valid_set)
        self.min_MdR_t2i = len(self.valid_set)
        self.best_epoch = 0
        self.checkpoint_dir = 'checkpoints/CLIP4H/MSRVTT'
        self.qB_img = self.qB_txt = self.rB_img = self.rB_txt = None

    def train(self):
        train_loader = DataLoader(self.train_set, batch_size=self.batch_size)
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
            self.valid(epoch)
            self.scheduler.step()

    def valid(self, epoch):
        #  1000 * bit_size
        qB_img, qB_txt = rB_img, rB_txt = get_valid_codes(self.valid_set, self.model, self.batch_size)
        gt_path = 'dataset/MSRVTT/ground_truth.txt'
        gt = np.loadtxt(gt_path, dtype=int).reshape(1000, 1000)
        gt = torch.tensor(gt).to('cuda')
        r1_i2t = calc_recalls_topn(1, qB_img, rB_txt, gt)
        r1_t2i = calc_recalls_topn(1, qB_txt, rB_img, gt)
        r5_i2t = calc_recalls_topn(5, qB_img, rB_txt, gt)
        r5_t2i = calc_recalls_topn(5, qB_txt, rB_img, gt)
        r10_i2t = calc_recalls_topn(10, qB_img, rB_txt, gt)
        r10_t2i = calc_recalls_topn(10, qB_txt, rB_img, gt)
        # MdR: median rank of GT videos
        MdR_i2t = calc_MdR(qB_img, rB_txt, gt)
        MdR_t2i = calc_MdR(qB_txt, rB_img, gt)

        if r1_i2t + r1_t2i + r5_i2t + r5_t2i + r10_i2t + r10_t2i - MdR_i2t - MdR_t2i >= \
                self.max_r1_i2t + self.max_r1_t2i + self.max_r5_i2t + self.max_r5_t2i + self.max_r10_i2t + self.max_r10_t2i \
                - self.min_MdR_i2t - self.min_MdR_t2i:
            self.max_r1_i2t = r1_i2t
            self.max_r1_t2i = r1_t2i
            self.max_r5_i2t = r5_i2t
            self.max_r5_t2i = r5_t2i
            self.max_r10_i2t = r10_i2t
            self.max_r10_t2i = r10_t2i
            self.min_MdR_i2t = MdR_i2t
            self.min_MdR_t2i = MdR_t2i
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), self.checkpoint_dir + '/CLIP4H-' + str(self.bit_size) + '-HashNet' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.pth')
            self.qB_img = qB_img.cpu()
            self.qB_txt = qB_txt.cpu()
            self.rB_img = rB_img.cpu()
            self.rB_txt = rB_txt.cpu()
        with open('log/log.txt', 'a') as f:
            line = f'epoch: [{epoch + 1}/{self.epoch}], R@1(i2t): {r1_i2t:.4f}, R@1(t2i): {r1_t2i}, R@5(i2t): {r5_i2t}, R@5(t2i): {r5_t2i}, R@10(i2t): {r10_i2t}, R@10(t2i): {r10_t2i}, MdR(i2t): {MdR_i2t}, MdR(t2i): {MdR_t2i}\n' \
                   f'best:  [{self.best_epoch + 1}/{self.epoch}], R@1(i2t): {self.max_r1_i2t:.4f}, R@1(t2i): {self.max_r1_t2i}, R@5(i2t): {self.max_r5_i2t}, R@5(t2i): {self.max_r5_t2i}, R@10(i2t): {self.max_r10_i2t}, R@10(t2i): {self.max_r10_t2i}, MdR(i2t): {self.min_MdR_i2t}, MdR(t2i): {self.min_MdR_t2i}\n'
            f.write(line)


def get_valid_codes(valid_set, HashNet_model, batch_size):
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    img_buffer = txt_buffer = torch.empty(len(valid_set), HashNet_model.bit_size, dtype=torch.float32)
    img_buffer, txt_buffer = to_cuda(img_buffer, txt_buffer)
    index = 0
    for image_feature, text_feature in tqdm(valid_loader):  # Fv and Ft
        F_I = image_feature.to('cuda', dtype=torch.float32)
        F_T = text_feature.to('cuda', dtype=torch.float32)
        hid_I, _ = HashNet_model(F_I)
        hid_T, _ = HashNet_model(F_T)
        for i in range(hid_I.shape[0]):
            img_buffer[index + i] = hid_I.data[i]
            txt_buffer[index + i] = hid_T.data[i]
        index = index + hid_I.shape[0]
    img_buffer = HashNet_model.get_B(img_buffer)
    txt_buffer = HashNet_model.get_B(txt_buffer)
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


def calc_recalls_topn(n, qB, rB, gt):
    # qB, rB: num_query * bit_size
    num_query = qB.shape[0]
    recalls = [0] * num_query
    for i in range(num_query):
        # qB[iter]'s hamm with every rB
        # hamm: 1*num_query
        hamm = calc_hammingDist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        #  found: the found right ones of iter
        found = gt[ind]
        # topn
        found = found[:n, i]

        # total: all right ones of iter
        total = torch.nonzero(gt[i]).squeeze(1)

        # recall = the found right ones / all right ones
        # found[: n] topn
        right = torch.nonzero(found).squeeze(1)
        recalls[i] += (right.numel()/total.numel())
    return sum(recalls) / len(recalls)


def calc_MdR(qB, rB, gt):
    # qB, rB: 1000*bit_size
    num_query = qB.shape[0]
    mdr = []
    for i in range(num_query):
        # qB[iter]'s hamm with every rB
        hamm = calc_hammingDist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        for rank in range(len(ind)):
            # ind[i] is a right one, save its rank
            if gt[i][ind[rank]] == 1:
                mdr.append(rank)
    median_rank = torch.tensor(mdr).median().item()
    return median_rank + 1


def calc_hammingDist(B1, B2):
    # B1: 1 * bit_size B2: 1000 * bit_size
    # q = bit_size
    q = B2.shape[1]
    # bit_size - 1's num = 0's num = distH
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = q - B1.mm(B2.transpose(0, 1))
    return distH


def train():
    trainer = CLIP4H()
    trainer.train()


def test():
    qB = torch.randint(0, 2, (10, 10))
    qB = qB.float() * 2 - 1
    rB = qB
    gt = torch.randint(0, 2, (10, 10))
    indices = torch.arange(0, 10)
    gt[indices, indices] = 1
    recall1 = calc_recalls_topn(1, qB, rB, gt)
    recall5 = calc_recalls_topn(5, qB, rB, gt)
    recall10 = calc_recalls_topn(10, qB, rB, gt)
    MdR = calc_MdR(qB, rB, gt)