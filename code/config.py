"""
系统配置文件 - Configuration Management
统一管理所有路径和参数配置
使用方法: from config import RESULTS_DIR, model_config
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# ==================== 项目路径配置 ====================

# 项目根目录(自动检测 - 基于config.py文件位置)
PROJECT_ROOT = Path(__file__).parent.resolve()

# 主要目录结构
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"
LOGS_DIR = PROJECT_ROOT / "logs"

# 结果子目录
SCENE_DETECT_DIR = RESULTS_DIR / "scene_detect"
SCENE_FRAMES_DIR = SCENE_DETECT_DIR / "frames"
SCENE_VIDEOS_DIR = SCENE_DETECT_DIR / "videos"
CLIP_SEARCH_DIR = RESULTS_DIR / "clip_search"
HYBRID_KEYFRAMES_DIR = RESULTS_DIR / "hybrid_keyframes"
HYBRID_VIDEOS_DIR = RESULTS_DIR / "hybrid_videos"

# 缓存子目录
FRAME_CACHE_DIR = CACHE_DIR / "frames"
MODEL_CACHE_DIR = CACHE_DIR / "models"

# 模型文件路径
TVSUM_MODEL_PATH = PROJECT_ROOT / "training" / "models" / "summarizer.pth"
CLIP_MODEL_DIR = MODELS_DIR / "chinese-clip"

# 数据目录
TEST_VIDEOS_DIR = DATA_DIR / "test_videos"
DATASETS_DIR = DATA_DIR / "datasets"
TVSUM_DATA_DIR = PROJECT_ROOT / "training" / "data" / "tvsum"
TVSUM_VIDEO_DIR = TVSUM_DATA_DIR / "video"
TVSUM_ANNOTATION_FILE = TVSUM_DATA_DIR / "ydata-tvsum50-anno.tsv"
TVSUM_INFO_FILE = TVSUM_DATA_DIR / "ydata-tvsum50-info.tsv"

# 论文实验目录
PAPER_EXPERIMENTS_DIR = PROJECT_ROOT / "paper_experiments"
PAPER_RESULTS_DIR = PAPER_EXPERIMENTS_DIR / "results"
PAPER_TMP_SUMMARY_DIR = PAPER_RESULTS_DIR / "_tmp_summaries"


# ==================== 模型配置 ====================

@dataclass
class ModelConfig:
    """模型配置类"""
    
    # LLM配置 (本地 Ollama)
    # Use the IPv4 loopback explicitly.  On some Windows installations
    # ``localhost`` resolves to ::1 while Ollama only listens on IPv4.
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1000
    
    # CLIP配置
    clip_model_name: str = "ViT-L-14-336"
    clip_batch_size: int = 32
    clip_download_root: str = str(MODELS_DIR)
    
    # TVSUM配置
    tvsum_model_path: str = str(TVSUM_MODEL_PATH)
    tvsum_sample_rate: int = 2
    tvsum_clip_length: int = 200
    
    def __post_init__(self):
        """从环境变量读取LLM配置(如果存在)"""
        env_base_url = os.environ.get("LLM_BASE_URL")
        env_api_key = os.environ.get("LLM_API_KEY")
        env_model = os.environ.get("LLM_MODEL")

        if env_base_url:
            self.llm_base_url = env_base_url
        if env_api_key:
            self.llm_api_key = env_api_key
        if env_model:
            self.llm_model = env_model


@dataclass
class ProcessConfig:
    """处理流程配置类"""
    
    # 场景检测参数
    scene_threshold: float = 27.0
    scene_min_len: int = 15
    scene_method: str = "content"
    
    # 视频输出参数
    output_fps: int = 5
    output_codec: str = "libx264"
    output_preset: str = "fast"
    output_crf: int = 23
    
    # 热力图参数
    heatmap_window_size: int = 128
    heatmap_stride: int = 32
    heatmap_alpha: float = 0.5
    
    # CLIP检索参数
    clip_default_top_k: int = 10
    clip_default_sample_fps: int = 1
    
    # 性能优化
    enable_cache: bool = True
    max_cache_size_gb: float = 5.0
    max_cache_age_days: int = 7
    
    # 调试模式
    debug: bool = False
    verbose: bool = False


# 全局配置实例
model_config = ModelConfig()
process_config = ProcessConfig()


# 目录初始化函数
def init_directories(verbose: bool = False):
    """初始化所有必要的目录结构"""
    directories = [
        RESULTS_DIR, MODELS_DIR, DATA_DIR, CACHE_DIR, LOGS_DIR,
        SCENE_DETECT_DIR, SCENE_FRAMES_DIR, SCENE_VIDEOS_DIR,
        CLIP_SEARCH_DIR, HYBRID_KEYFRAMES_DIR, HYBRID_VIDEOS_DIR,
        FRAME_CACHE_DIR, MODEL_CACHE_DIR,
        TEST_VIDEOS_DIR, DATASETS_DIR,
        TVSUM_DATA_DIR, TVSUM_VIDEO_DIR,
        PAPER_RESULTS_DIR, PAPER_TMP_SUMMARY_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("[OK] 目录结构初始化完成")


# 辅助函数
def get_video_output_path(video_name: str, suffix: str = "summary") -> Path:
    """生成视频输出路径"""
    base_name = Path(video_name).stem
    return HYBRID_VIDEOS_DIR / f"{base_name}_{suffix}.mp4"


def get_scene_output_path(video_name: str) -> Path:
    """生成场景检测视频输出路径"""
    base_name = Path(video_name).stem
    return SCENE_VIDEOS_DIR / f"{base_name}_scene_summary.mp4"


# 自动初始化
if not RESULTS_DIR.exists():
    init_directories(verbose=False)


if __name__ == "__main__":
    init_directories(verbose=True)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"结果目录: {RESULTS_DIR}")
