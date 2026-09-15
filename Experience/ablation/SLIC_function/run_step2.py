"""
第二步：单张切片的节点特征提取 - 入口脚本

用法:
  python run_step2.py
  python run_step2.py --step1_output ./output/step1 --output ./output/step2
  python run_step2.py --image_dir /path/to/images  # 若有原始图像，可提取纹理等特征
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.run_step2 import main

if __name__ == "__main__":
    main()
