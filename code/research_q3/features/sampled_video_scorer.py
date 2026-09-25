"""Score arbitrary sampled video frames with Visual-BiLSTM and Chinese-CLIP."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image

from config import TVSUM_MODEL_PATH
from tools.clip_model_factory import get_clip_model
from training.dataset import Config
from training.model import VideoSummarizer


class SampledVideoScorer:
    """Loads the two frozen encoders once and scores sampled frames in batches."""

    def __init__(self, device: str | None = None, batch_size: int = 32):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = int(batch_size)
        self.config = Config()
        self.visual_model = VideoSummarizer(self.config).to(self.device)
        state = torch.load(TVSUM_MODEL_PATH, map_location=self.device)
        self.visual_model.load_state_dict(state)
        self.visual_model.eval()
        self.clip_model, self.clip_preprocess, self.clip_module = get_clip_model(
            device=str(self.device)
        )

    @staticmethod
    def read_sampled_rgb(video_path: str | Path, frame_indices: Iterable[int]) -> list[np.ndarray]:
        """Read sorted target frames in one sequential pass through the video."""
        targets = np.asarray(sorted(set(int(i) for i in frame_indices)), dtype=np.int64)
        if targets.size == 0:
            return []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"cannot open video: {video_path}")
        frames: list[np.ndarray] = []
        target_pos = 0
        frame_pos = 0
        while target_pos < targets.size:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_pos == targets[target_pos]:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                target_pos += 1
            frame_pos += 1
        cap.release()
        if len(frames) != targets.size:
            raise RuntimeError(
                f"read {len(frames)}/{targets.size} requested frames from {video_path}"
            )
        return frames

    def _visual_features(self, rgb_frames: list[np.ndarray]) -> torch.Tensor:
        batches = []
        with torch.no_grad():
            for start in range(0, len(rgb_frames), self.batch_size):
                tensors = [self.config_transform(frame) for frame in rgb_frames[start : start + self.batch_size]]
                batch = torch.stack(tensors).to(self.device)
                features = self.visual_model.encoder(batch).flatten(1)
                batches.append(features.cpu())
        return torch.cat(batches, dim=0)

    def config_transform(self, rgb_frame: np.ndarray) -> torch.Tensor:
        """Apply exactly the transform used to train Visual-BiLSTM."""
        from training.dataset import VideoProcessor

        if not hasattr(self, "_visual_transform"):
            self._visual_transform = VideoProcessor(self.config).transform
        return self._visual_transform(rgb_frame)

    def score_visual(self, rgb_frames: list[np.ndarray]) -> np.ndarray:
        features = self._visual_features(rgb_frames).unsqueeze(0).to(self.device)
        with torch.no_grad():
            sequence, _ = self.visual_model.lstm(features)
            scores = self.visual_model.scorer(sequence).squeeze(0).squeeze(-1)
        return scores.detach().cpu().numpy().astype(np.float64)

    def score_semantic(self, rgb_frames: list[np.ndarray], text_query: str) -> np.ndarray:
        clip = self.clip_module
        with torch.no_grad():
            text = clip.tokenize([text_query]).to(self.device)
            text_features = self.clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(rgb_frames), self.batch_size):
                images = [
                    self.clip_preprocess(Image.fromarray(frame))
                    for frame in rgb_frames[start : start + self.batch_size]
                ]
                image_tensor = torch.stack(images).to(self.device)
                image_features = self.clip_model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                values = (image_features @ text_features.T).squeeze(-1)
                scores.extend(values.detach().cpu().numpy().astype(float).tolist())
        return np.asarray(scores, dtype=np.float64)

    def encode_clip_images(self, rgb_frames: list[np.ndarray]) -> np.ndarray:
        """Return normalized CLIP image embeddings for a frame batch."""
        features = []
        with torch.no_grad():
            for start in range(0, len(rgb_frames), self.batch_size):
                images = [
                    self.clip_preprocess(Image.fromarray(frame))
                    for frame in rgb_frames[start : start + self.batch_size]
                ]
                tensor = torch.stack(images).to(self.device)
                encoded = self.clip_model.encode_image(tensor)
                encoded = encoded / encoded.norm(dim=-1, keepdim=True)
                features.append(encoded.detach().cpu())
        return torch.cat(features).numpy().astype(np.float32)

    def encode_clip_texts(self, queries: list[str]) -> np.ndarray:
        """Return normalized CLIP text embeddings for frozen query strings."""
        with torch.no_grad():
            tokens = self.clip_module.tokenize(queries).to(self.device)
            encoded = self.clip_model.encode_text(tokens)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
        return encoded.detach().cpu().numpy().astype(np.float32)

    def score_image_paths(self, image_paths: Iterable[str | Path]) -> dict[str, np.ndarray]:
        """Cache visual scores and CLIP image embeddings without retaining all images."""
        paths = [Path(path) for path in image_paths]
        visual_chunks: list[torch.Tensor] = []
        clip_chunks: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            frames = []
            for path in paths[start : start + self.batch_size]:
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise OSError(f"cannot read image: {path}")
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            visual_chunks.append(self._visual_features(frames))
            clip_chunks.append(self.encode_clip_images(frames))
        visual_features = torch.cat(visual_chunks, dim=0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            sequence, _ = self.visual_model.lstm(visual_features)
            visual_scores = self.visual_model.scorer(sequence).squeeze(0).squeeze(-1)
        return {
            "visual_scores": visual_scores.detach().cpu().numpy().astype(np.float64),
            "clip_image_features": np.concatenate(clip_chunks, axis=0),
        }

    def score_video(
        self,
        video_path: str | Path,
        frame_indices: Iterable[int],
        text_query: str,
    ) -> dict[str, np.ndarray]:
        indices = np.asarray(sorted(set(int(i) for i in frame_indices)), dtype=np.int64)
        frames = self.read_sampled_rgb(video_path, indices)
        return {
            "indices": indices,
            "visual_scores": self.score_visual(frames),
            "semantic_scores": self.score_semantic(frames, text_query),
        }
