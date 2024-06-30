from torch import nn
import torch


def get_B(U):  # U: batch_size * bit_size
    temp_U = U.t()  # temp_U: every column represents a bit
    max_value, max_index = torch.max(temp_U, dim=1)  # torch.max: dim=1 means every row's max
    min_value, min_index = torch.min(temp_U, dim=1)  # find max and min in a batch, 256
    maxmin_values = torch.stack([max_value, min_value], dim=1).unsqueeze(2)  # 256 * 2 * 1
    temp_U = temp_U.unsqueeze(2)  # 256 * 3 * 1
    dist = torch.cdist(temp_U, maxmin_values, p=1).to('cuda')  # calculate distance, p = 1 曼哈顿距离: d = |x1 - x2| + |y1 - y2|
    # 当x1和x2的形状分别为[B, P, M]和[B, R, M]时，torch.cdist的结果形状为[B, P, R] 256 3 1, 256 2 1: 256 3 2, 256bit 每个bit有3个hid，每个hid有2个结果，to_max_dis, to_min_dis
    differences = dist[:, :, 0] - dist[:, :, 1]  # to_max_dis - to_min_dis
    B = torch.zeros(differences.shape).to('cuda')
    B[differences <= 0] = 1
    B[differences > 0] = -1
    B = B.t().to('cuda')
    return B


class HashNet(nn.Module):
    def __init__(self, bit_size):
        super(HashNet, self).__init__()
        self.bit_size = bit_size
        self.fc1 = nn.Linear(512, 4096)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        # use ReLU in MLP, then use min-max hashing layer to get hashcode
        self.fc_encode = nn.Linear(4096, bit_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        feature = x
        hid = self.fc_encode(feature)  # Hz z = bit_size
        code = get_B(hid)
        return hid, code


def get_model(bit_size):
    return HashNet(bit_size)
