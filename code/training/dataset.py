import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from config import RESULTS_DIR, TVSUM_MODEL_PATH, TVSUM_VIDEO_DIR, TVSUM_ANNOTATION_FILE


class Config:
    def __init__(self):
        self.dataset_name = "TVSum"
        self.output_dir = str(RESULTS_DIR)
        self.model_save_path = str(TVSUM_MODEL_PATH)

        # 数据集路径
        self.video_dir = str(TVSUM_VIDEO_DIR)
        self.annotation_file = str(TVSUM_ANNOTATION_FILE)

        # 数据处理参数
        self.sample_rate = 5  # 每秒采样的帧数
        self.clip_length = 90  # 视频最大长度（秒）
        self.img_size = 160  # 图像大小

        # 模型参数
        self.hidden_dim = 512
        self.num_layers = 2
        self.dropout = 0.5
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.batch_size = 1
        self.epochs = 50

        # 创建必要的目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)


class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_frames(self, video_path):
        """从视频文件中提取帧"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 检查视频是否成功打开
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return None

        # 计算采样间隔
        stride = int(fps / self.config.sample_rate)
        if stride == 0:
            stride = 1  # 确保至少每帧采样一次

        frame_count = 0
        success = True

        max_frames = self.config.clip_length * self.config.sample_rate

        while success and len(frames) < max_frames:
            success, frame = cap.read()
            if not success:
                break

            if frame_count % stride == 0:
                # 将BGR转换为RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 应用转换
                frame_tensor = self.transform(frame)
                frames.append(frame_tensor)

            frame_count += 1

        cap.release()

        # 确保帧数统一
        if len(frames) == 0:
            print(f"没有从视频中提取到帧: {video_path}")
            return None

        # 将帧堆叠成张量
        frames_tensor = torch.stack(frames)
        return frames_tensor

    def save_summary(self, video_path, selected_indices, output_path):
        """生成并保存视频摘要"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 计算采样间隔
        stride = int(fps / self.config.sample_rate)
        if stride == 0:
            stride = 1  # 确保至少每帧采样一次

        frame_count = 0
        success = True

        while success:
            success, frame = cap.read()
            if not success:
                break

            # 检查当前帧是否在选定的索引中
            sampled_index = frame_count // stride
            if sampled_index in selected_indices:
                out.write(frame)

            frame_count += 1

        cap.release()
        out.release()


class TVSumDataset(Dataset):
    def __init__(self, config, video_list, annotation_data):
        self.config = config
        self.video_list = video_list
        self.annotation_data = annotation_data
        self.processor = VideoProcessor(config)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_info = self.video_list[idx]
        video_path = video_info['path']
        video_id = video_info['id']

        # 提取帧
        frames = self.processor.extract_frames(video_path)

        # 获取标注数据
        if video_id in self.annotation_data:
            scores = self.annotation_data[video_id]
            # 调整scores长度以匹配帧数
            if frames is not None:
                num_frames = len(frames)
                num_scores = len(scores)
                if num_frames != num_scores:
                    # 插值调整
                    indices = np.linspace(0, num_scores - 1, num_frames)
                    scores = np.interp(indices, np.arange(num_scores), scores)
        else:
            # 如果没有标注，生成随机分数（仅用于推理）
            if frames is not None:
                scores = np.random.rand(len(frames))
            else:
                scores = np.array([])

        scores_tensor = torch.FloatTensor(scores)

        return {
            'frames': frames,
            'scores': scores_tensor,
            'video_id': video_id
        }


def load_tvsum_annotations(annotation_file):
    """加载TVSum数据集的标注"""
    annotation_data = {}

    try:
        df = pd.read_csv(annotation_file, sep='\t')

        # 处理每个视频的标注
        for video_id in df['video'].unique():
            video_data = df[df['video'] == video_id]

            # 收集所有标注者的分数
            all_scores = []
            for _, row in video_data.iterrows():
                scores = [float(x) for x in row['annotations'].split(',')]
                all_scores.append(scores)

            # 取平均分数作为ground truth
            avg_scores = np.mean(all_scores, axis=0)
            # 归一化到[0, 1]
            avg_scores = (avg_scores - avg_scores.min()) / (avg_scores.max() - avg_scores.min() + 1e-8)

            annotation_data[video_id] = avg_scores
    except Exception as e:
        print(f"加载标注文件时出错: {e}")

    return annotation_data


def get_video_list(video_dir):
    """获取视频文件列表"""
    video_list = []

    if os.path.exists(video_dir):
        for video_file in os.listdir(video_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(video_dir, video_file)
                video_id = os.path.splitext(video_file)[0]
                video_list.append({
                    'path': video_path,
                    'id': video_id,
                    'filename': video_file
                })
    else:
        print(f"视频目录不存在: {video_dir}")

    return video_list
