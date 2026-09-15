"""
第三步：单张切片的图神经网络编码 - 入口脚本

用法:
  python run_step3.py
  python run_step3.py --step2_output ./output/step2 --output ./output/step3
  python run_step3.py --patients 10105856 --d_slice 64
  python run_step3.py --ckpt ./ckpt/gnn.pth  # 可选：加载预训练 GNN
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.run_step3 import main

if __name__ == "__main__":
    main()
