import torch
import torch.nn as nn
from torchvision import models


class VideoSummarizer(nn.Module):
    def __init__(self, config):
        super(VideoSummarizer, self).__init__()
        self.config = config

        # 使用预训练的ResNet作为特征提取器
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的全连接层

        # 冻结特征提取器参数
        for param in self.encoder.parameters():
            param.requires_grad = False

        # BiLSTM用于序列建模
        self.lstm = nn.LSTM(
            input_size=2048,  # ResNet50的输出维度
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,  # 批次维度在前
            bidirectional=True,  # 双向LSTM
            dropout=config.dropout
        )

        # 重要性分数预测器
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, 128),  # *2因为是双向LSTM
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 使用Sigmoid确保输出在[0, 1]范围内
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape

        # 重塑张量以便于特征提取
        x = x.view(batch_size * seq_len, c, h, w)

        # 提取特征
        with torch.no_grad():  # 不计算梯度，节省内存
            features = self.encoder(x)

        # 重塑特征
        features = features.view(batch_size, seq_len, -1)

        # 序列建模
        lstm_out, _ = self.lstm(features)

        # 预测重要性分数
        scores = self.scorer(lstm_out).squeeze(-1)  # [batch_size, seq_len]

        return scores