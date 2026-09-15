"""
第五步：临床数据读取与模态对齐 - 入口脚本

用法:
  python run_step5.py
  python run_step5.py --clinical ./clinical.csv --step4_output ./output/step4
  python run_step5.py --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.run_step5 import main

if __name__ == "__main__":
    main()
