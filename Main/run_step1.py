"""
第一步：单张切片的图构建 - 入口脚本

用法:
  python run_step1.py
  python run_step1.py --dataset ./dataset --output ./output/step1
  python run_step1.py --image_dir /path/to/images  # 若有原始图像，可提取像素特征
"""

import sys
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.run_step1 import main

if __name__ == "__main__":
    main()
