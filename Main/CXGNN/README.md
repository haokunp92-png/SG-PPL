# CXGNN-main (minimal)

仅保留 sPixel_CXGNN 流水线所需的 NNModel 编码器。

- `model/alg1.py`: NNModel 类（共享 MLP 编码器）
- 流水线 step6 通过 `from model import alg1` 获取 `alg1.NNModel`
