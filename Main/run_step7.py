"""
第七步：绘制生存分析图像 - 入口脚本

用法:
  python run_step7.py --patients 10071200
  python run_step7.py --patient_indices 0
  python run_step7.py --patients 10071200,10093300 --output ./output/step7
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.run_step7 import main

if __name__ == "__main__":
    main()
