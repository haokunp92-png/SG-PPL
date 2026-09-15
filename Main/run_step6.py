"""
第六步：因果模型构建与反事实推理 - 入口脚本

用法:
  python run_step6.py
  python run_step6.py --step5_output ./output/step5 --output ./output/step6
  python run_step6.py --epochs 200 --balance_reg 0.1
  python run_step6.py --inference_all   # 对全部患者进行反事实推理
  python run_step6.py --max_patients 200 --output ./output/step6_n200  # 仅用 aligned 顺序前 200 人
  # 验证集：默认按 C-index 存最优权重（见 step6_causal_model）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.run_step6 import main

if __name__ == "__main__":
    main()
