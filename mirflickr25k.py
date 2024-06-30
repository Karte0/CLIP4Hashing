# -*- coding: utf-8 -*-
# @Time    : 2019/6/27
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from torchcmh.dataset.base import *
from scipy import io as sio
import numpy as np
from torchcmh.dataset import abs_dir
import os
import torchcmh.models.pretrain_model.pretrained_Bert.bert_for_classification.Tasks.TaskForPairSentenceClassification as bert_for_classification
from torchcmh.models.pretrain_model.pretrained_Bert.bert_for_classification.utils import LoadPairSentenceClassificationDataset
from torchcmh.models.pretrain_model.pretrained_Bert.bert_for_classification.Tasks.TaskForPairSentenceClassification import ModelConfig

default_img_mat_url = os.path.join(abs_dir,
                                   "../../../../files/大四下/毕设/DCMH-0407/deep-cross-modal-hashing-master/torchcmh/dataset/data", "mirflickr25k", "imgList.mat")
default_tag_mat_url = os.path.join(abs_dir,
                                   "../../../../files/大四下/毕设/DCMH-0407/deep-cross-modal-hashing-master/torchcmh/dataset/data", "mirflickr25k", "tagList.mat")
default_label_mat_url = os.path.join(abs_dir,
                                     "../../../../files/大四下/毕设/DCMH-0407/deep-cross-modal-hashing-master/torchcmh/dataset/data", "mirflickr25k", "labelList.mat")
default_seed = 6


img_names = None
txt = None
label = None


def load_mat(img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url, label_mat_url=default_label_mat_url):
    img_names = sio.loadmat(img_mat_url)['FAll']  # type: np.ndarray
    img_names = img_names.squeeze()
    all_img_names = np.array([name[0] for name in img_names])
    all_txt = np.array(sio.loadmat(tag_mat_url)['YAll'])
    all_label = np.array(sio.loadmat(label_mat_url)['LAll'])
    return all_img_names, all_txt, all_label


def split_data(all_img_names, all_txt, all_label, query_num=2000, train_num=10000, seed=None):
    np.random.seed(seed)
    random_index = np.random.permutation(all_txt.shape[0])
    query_index = random_index[: query_num]
    train_index = random_index[query_num: query_num + train_num]
    retrieval_index = random_index[query_num:]

    query_img_names = all_img_names[query_index]
    train_img_names = all_img_names[train_index]
    retrieval_img_names = all_img_names[retrieval_index]

    query_txt = all_txt[query_index]
    train_txt = all_txt[train_index]
    retrieval_txt = all_txt[retrieval_index]

    query_label = all_label[query_index]
    train_label = all_label[train_index]
    retrieval_label = all_label[retrieval_index]

    img_names = (query_img_names, train_img_names, retrieval_img_names)
    txt = (query_txt, train_txt, retrieval_txt)
    label = (query_label, train_label, retrieval_label)
    return img_names, txt, label


def load_dataset(train_dataset, img_dir, img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url,
                  label_mat_url=default_label_mat_url, batch_size=32, train_num=10000, query_num=2000, seed=default_seed, **kwargs):
    global img_names, txt, label
    if img_names is None:
        all_img_names, all_txt, all_label = load_mat(img_mat_url, tag_mat_url, label_mat_url)
        img_names, txt, label = split_data(all_img_names, all_txt, all_label, query_num, train_num, seed)
        print("mirflickr25k data load and shuffle by seed %d" % seed)
    img_train_transform = kwargs['img_train_transform'] if 'img_train_transform' in kwargs.keys() else None
    txt_train_transform = kwargs['txt_train_transform'] if 'txt_train_transform' in kwargs.keys() else None
    img_valid_transform = kwargs['img_valid_transform'] if 'img_valid_transform' in kwargs.keys() else None
    txt_valid_transform = kwargs['txt_valid_transform'] if 'txt_valid_transform' in kwargs.keys() else None
    train_data = train_dataset(img_dir, img_names[1], txt[1], label[1], batch_size, img_train_transform, txt_train_transform)
    valid_data = CrossModalValidBase(img_dir, img_names[0], img_names[2], txt[0], txt[2], label[0], label[2], img_valid_transform,
                                     txt_valid_transform)
    return train_data, valid_data

# def load_dataset(train_dataset, img_dir, img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url,
#                  label_mat_url=default_label_mat_url, batch_size=32, train_num=10000, query_num=2000, seed=default_seed, **kwargs):
#     global img_names, txt, label
#     if img_names is None:
#         all_img_names, all_txt, all_label = load_mat(img_mat_url, tag_mat_url, label_mat_url)
#         img_names, txt, label = split_data(all_img_names, all_txt, all_label, query_num, train_num, seed)
#         print("mirflickr25k data load and shuffle by seed %d" % seed)
#     img_train_transform = kwargs['img_train_transform'] if 'img_train_transform' in kwargs.keys() else None
#     txt_train_transform = kwargs['txt_train_transform'] if 'txt_train_transform' in kwargs.keys() else None
#     img_valid_transform = kwargs['img_valid_transform'] if 'img_valid_transform' in kwargs.keys() else None
#     txt_valid_transform = kwargs['txt_valid_transform'] if 'txt_valid_transform' in kwargs.keys() else None
#     # ids = []
#     # for img_name in all_img_names:
#     #     ids.append(img_name.split('im')[1].split('.jpg')[0])
#     # dir = 'D:/files/大四下/毕设/DCMH-0407/deep-cross-modal-hashing-master//torchcmh/dataset/data/mirflickr25k/mirflickr/meta/tags'
#     # lst = os.listdir(dir)
#     # txt_content = {}
#     # for tag in lst:
#     #     with open(dir + '/' + tag, "r", encoding='utf-8') as f:
#     #         content = ""
#     #         for line in f.readlines():
#     #             content = content + line.strip() + " "
#     #         txt_content['img' + tag.split('tags')[1].split('.txt')[0]] = content
#
#
#     all_txt_content = {}
#     #file = open('D:/files/大四下/毕设/DCMH-0407/deep-cross-modal-hashing-master//torchcmh/dataset/mirflickr25k_content.txt', "r", encoding='utf-8')
#     #file2 = open('D:/files/大四下/毕设/DCMH-0407/deep-cross-modal-hashing-master//torchcmh/dataset/mirflickr25k_img_ids.txt', "r", encoding='utf-8')
#     file = open('/home/team/liuyansong/deep-cross-modal-hashing-master/torchcmh/dataset/mirflickr25k_content.txt', "r", encoding='utf-8')
#     file2 = open('/home/team/liuyansong/deep-cross-modal-hashing-master/torchcmh/dataset/mirflickr25k_img_ids.txt', "r", encoding='utf-8')
#
#     for line, line2 in zip(file.readlines(), file2.readlines()):
#         all_txt_content[line2.split('\n')[0]] = line.split('\n')[0]
#     file.close()
#     file2.close()
#
#
#     txt_content_train = []
#     for names in img_names[1]:
#         txt_content_train.append(all_txt_content[names.split('.jpg')[0]])
#
#     txt_content_valid_0 = []
#     for names in img_names[0]:
#         txt_content_valid_0.append(all_txt_content[names.split('.jpg')[0]])
#
#     txt_content_valid_2 = []
#     for names in img_names[2]:
#         txt_content_valid_2.append(all_txt_content[names.split('.jpg')[0]])
#
#
#     tokenizer = bert_for_classification.get_bert_tokenizer()
#     config = ModelConfig()
#     data_loader = LoadPairSentenceClassificationDataset(
#         vocab_path=config.vocab_path,
#         tokenizer=tokenizer,
#         batch_size=config.batch_size,
#         max_sen_len=config.max_sen_len,
#         split_sep=config.split_sep,
#         max_position_embeddings=config.max_position_embeddings,
#         pad_index=config.pad_token_id)
#     # labels = ['animals', 'baby', 'bird', 'car', 'clouds', 'dog', 'female', 'flower', 'food', 'indoor', 'lake', 'male', 'night', 'people', 'plant_life', 'portrait', 'river', 'sea', 'sky', 'structures', 'sunset', 'transport', 'tree', 'water']
#     # label_content_1 = []
#     # label_content_0 = []
#     # label_content_2 = []
#     # for l in label[1]:
#     #     str = ""
#     #     for index, j in enumerate(l):
#     #         if j == 1:
#     #             str = str + labels[index] + " "
#     #     label_content_1.append(str)
#     # for l in label[0]:
#     #     str = ""
#     #     for index, j in enumerate(l):
#     #         if j == 1:
#     #             str = str + labels[index] + " "
#     #     label_content_0.append(str)
#     # for l in label[2]:
#     #     str = ""
#     #     for index, j in enumerate(l):
#     #         if j == 1:
#     #             str = str + labels[index] + " "
#     #     label_content_2.append(str)
#
#     # txt_content_train = data_loader.data_process(txt_content_train, label[1])
#     # txt_content_valid_0 = data_loader.data_process(txt_content_valid_0, label[0])
#     # txt_content_valid_2 = data_loader.data_process(txt_content_valid_2, label[2])
#     txt_content_train = data_loader.data_process(txt_content_train)
#     txt_content_valid_0 = data_loader.data_process(txt_content_valid_0)
#     txt_content_valid_2 = data_loader.data_process(txt_content_valid_2)
#
#     tct = data_loader.generate_batch(txt_content_train).t().float()
#     tcv0 = data_loader.generate_batch(txt_content_valid_0).t().float()
#     tcv2 = data_loader.generate_batch(txt_content_valid_2).t().float()
#
#     train_data = train_dataset(img_dir, img_names[1], tct, label[1], batch_size, img_train_transform, txt_train_transform)
#     valid_data = CrossModalValidBase(img_dir, img_names[0], img_names[2], tcv0, tcv2, label[0], label[2], img_valid_transform,
#                                      txt_valid_transform)
#     return train_data, valid_data


def get_single_datasets(img_dir, img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url, label_mat_url=default_label_mat_url,
                        batch_size=32, train_num=10000, query_num=2000, seed=default_seed, **kwargs):
    print("load data set single mirflickr25k")
    return load_dataset(CrossModalSingleTrain, img_dir, img_mat_url, tag_mat_url, label_mat_url, batch_size=batch_size, train_num=train_num,
                        query_num=query_num, seed=seed, **kwargs)


def get_pairwise_datasets(img_dir, img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url, label_mat_url=default_label_mat_url,
                          batch_size=2,train_num=10000, query_num=2000, seed=default_seed, **kwargs):
    print("load data set pairwise mirflickr25k")
    return load_dataset(CrossModalPairwiseTrain, img_dir, img_mat_url, tag_mat_url, label_mat_url, batch_size=batch_size,
                        train_num=train_num, query_num=query_num, seed=seed, **kwargs)


def get_triplet_datasets(img_dir, img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url, label_mat_url=default_label_mat_url,
                         batch_size=2, train_num=10000, query_num=2000, seed=default_seed, **kwargs):
    print("load data set triplet mirflickr25k")
    return load_dataset(CrossModalTripletTrain, img_dir, img_mat_url, tag_mat_url, label_mat_url, batch_size=batch_size,
                        train_num=train_num, query_num=query_num, seed=seed, **kwargs)


def get_quadruplet_datasets(img_dir, img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url, label_mat_url=default_label_mat_url,
                            batch_size=2, train_num=10000, query_num=2000, seed=default_seed, **kwargs):
    print("load data set quadruplet mirflickr25k")
    return load_dataset(CrossModalQuadrupletTrain, img_dir, img_mat_url, tag_mat_url, label_mat_url, batch_size=batch_size,
                        train_num=train_num, query_num=query_num, seed=seed, **kwargs)
# #
# #
# #
# # -*- coding: utf-8 -*-
# # @Time    : 2019/6/27
# # @Author  : Godder
# # @Github  : https://github.com/WangGodder
# from .base import *
# from scipy import io as sio
# import numpy as np
# from torchcmh.dataset import abs_dir
# import os
#
# default_img_mat_url = os.path.join(abs_dir, "data", "mirflickr25k", "imgList.mat")
# default_tag_mat_url = os.path.join(abs_dir, "data", "mirflickr25k", "tagList.mat")
# default_label_mat_url = os.path.join(abs_dir, "data", "mirflickr25k", "labelList.mat")
# default_seed = 6
#
#
# img_names = None
# txt = None
# label = None
#
#
# def load_mat(img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url, label_mat_url=default_label_mat_url):
#     img_names = sio.loadmat(img_mat_url)['FAll']  # type: np.ndarray
#     img_names = img_names.squeeze()
#     all_img_names = np.array([name[0] for name in img_names])
#     all_txt = np.array(sio.loadmat(tag_mat_url)['YAll'])
#     all_label = np.array(sio.loadmat(label_mat_url)['LAll'])
#     return all_img_names, all_txt, all_label
#
#
# def split_data(all_img_names, all_txt, all_label, query_num=2000, train_num=10000, seed=None):
#     np.random.seed(seed)
#     random_index = np.random.permutation(all_txt.shape[0])
#     query_index = random_index[: query_num]
#     train_index = random_index[query_num: query_num + train_num]
#     retrieval_index = random_index[query_num:]
#
#     query_img_names = all_img_names[query_index]
#     train_img_names = all_img_names[train_index]
#     retrieval_img_names = all_img_names[retrieval_index]
#
#     query_txt = all_txt[query_index]
#     train_txt = all_txt[train_index]
#     retrieval_txt = all_txt[retrieval_index]
#
#     query_label = all_label[query_index]
#     train_label = all_label[train_index]
#     retrieval_label = all_label[retrieval_index]
#
#     img_names = (query_img_names, train_img_names, retrieval_img_names)
#     txt = (query_txt, train_txt, retrieval_txt)
#     label = (query_label, train_label, retrieval_label)
#     return img_names, txt, label
#
#
# def load_dataset(train_dataset, img_dir, img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url,
#                  label_mat_url=default_label_mat_url, batch_size=128, train_num=10000, query_num=2000, seed=default_seed, **kwargs):
#     global img_names, txt, label
#     if img_names is None:
#         all_img_names, all_txt, all_label = load_mat(img_mat_url, tag_mat_url, label_mat_url)
#         img_names, txt, label = split_data(all_img_names, all_txt, all_label, query_num, train_num, seed)
#         print("mirflickr25k data load and shuffle by seed %d" % seed)
#     img_train_transform = kwargs['img_train_transform'] if 'img_train_transform' in kwargs.keys() else None
#     txt_train_transform = kwargs['txt_train_transform'] if 'txt_train_transform' in kwargs.keys() else None
#     img_valid_transform = kwargs['img_valid_transform'] if 'img_valid_transform' in kwargs.keys() else None
#     txt_valid_transform = kwargs['txt_valid_transform'] if 'txt_valid_transform' in kwargs.keys() else None
#     train_data = train_dataset(img_dir, img_names[1], txt[1], label[1], batch_size, img_train_transform, txt_train_transform)
#     valid_data = CrossModalValidBase(img_dir, img_names[0], img_names[2], txt[0], txt[2], label[0], label[2], img_valid_transform,
#                                      txt_valid_transform)
#     return train_data, valid_data
#
#
# def get_single_datasets(img_dir, img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url, label_mat_url=default_label_mat_url,
#                         batch_size=128, train_num=10000, query_num=2000, seed=default_seed, **kwargs):
#     print("load data set single mirflickr25k")
#     return load_dataset(CrossModalSingleTrain, img_dir, img_mat_url, tag_mat_url, label_mat_url, batch_size=batch_size, train_num=train_num,
#                         query_num=query_num, seed=seed, **kwargs)
#
#
# def get_pairwise_datasets(img_dir, img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url, label_mat_url=default_label_mat_url,
#                           batch_size=128, train_num=10000, query_num=2000, seed=default_seed, **kwargs):
#     print("load data set pairwise mirflickr25k")
#     return load_dataset(CrossModalPairwiseTrain, img_dir, img_mat_url, tag_mat_url, label_mat_url, batch_size=batch_size,
#                         train_num=train_num, query_num=query_num, seed=seed, **kwargs)
#
#
# def get_triplet_datasets(img_dir, img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url, label_mat_url=default_label_mat_url,
#                          batch_size=128, train_num=10000, query_num=2000, seed=default_seed, **kwargs):
#     print("load data set triplet mirflickr25k")
#     return load_dataset(CrossModalTripletTrain, img_dir, img_mat_url, tag_mat_url, label_mat_url, batch_size=batch_size,
#                         train_num=train_num, query_num=query_num, seed=seed, **kwargs)
#
#
# def get_quadruplet_datasets(img_dir, img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url, label_mat_url=default_label_mat_url,
#                             batch_size=128, train_num=10000, query_num=2000, seed=default_seed, **kwargs):
#     print("load data set quadruplet mirflickr25k")
#     return load_dataset(CrossModalQuadrupletTrain, img_dir, img_mat_url, tag_mat_url, label_mat_url, batch_size=batch_size,
#                         train_num=train_num, query_num=query_num, seed=seed, **kwargs)
#
#
# __all__ = ['get_single_datasets', 'get_pairwise_datasets', 'get_triplet_datasets', 'get_quadruplet_datasets']
