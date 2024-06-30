import cv2
import os
from tqdm import tqdm

# 6 frames for MSVD,12 frames for MSRVTT,and 24 frames for DiDeMo
# 设置视频被分割后的总帧数
num = 12

# 视频文件夹路径
video_folder_path_train = 'dataset/MSRVTT/train-video/train-video'
video_folder_path_test = 'dataset/MSRVTT/test-video/test-video'

# 遍历视频文件夹中的所有文件
for filename in os.listdir(video_folder_path_train):
    # 构建完整的文件路径
    file_path = os.path.join(video_folder_path_train, filename)

    # 检查是否为视频文件
    if file_path.endswith(('.mp4', '.avi', '.mov')):
        # 打开视频文件
        cap = cv2.VideoCapture(file_path)
        # 获取总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 计算间隔
        interval = total_frames // num
        # 初始化进度条
        pbar = tqdm(total=num, desc="Caputring frames")
        # 循环截取帧
        for i in range(num):
            frame_id = i * interval
            # 定位帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            # 读取帧
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(
                    f"dataset/MSRVTT/train-video/train-video-frame/{filename.replace('.mp4', '')}frame{i}.jpg", frame)
                pbar.update(1)

        pbar.close()
        cap.release()
for filename in os.listdir(video_folder_path_test):
    # 构建完整的文件路径
    file_path = os.path.join(video_folder_path_test, filename)

    # 检查是否为视频文件
    if file_path.endswith(('.mp4', '.avi', '.mov')):
        # 打开视频文件
        cap = cv2.VideoCapture(file_path)
        # 获取总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 计算间隔
        interval = total_frames // num
        # 初始化进度条
        pbar = tqdm(total=num, desc="Caputring frames")
        # 循环截取帧
        for i in range(num):
            frame_id = i * interval
            # 定位帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            # 读取帧
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(
                    f"dataset/MSRVTT/test-video/test-video-frame/{filename.replace('.mp4', '')}frame{i}.jpg", frame)
                pbar.update(1)

        pbar.close()
        cap.release()