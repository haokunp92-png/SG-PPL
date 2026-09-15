"""
第四步：多切片融合为患者级图像特征 - 入口脚本

用法:
  python run_step4.py
  python run_step4.py --step3_output ./output/step3 --output ./output/step4
  python run_step4.py --patients 10105856
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.run_step4 import main

if __name__ == "__main__":
    main()
