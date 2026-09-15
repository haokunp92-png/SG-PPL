import torch
import os
from spixel_net import SpixelNet  # 导入你的模型类


def export_model():
    # 1. 初始化模型
    # 这里根据你的需求选择是否使用 BatchNorm (batchNorm=True 或 False)
    model = SpixelNet(batchNorm=True)

    # 2. 加载权重（可选）
    # 如果你有训练好的 .tar 或 .pth 文件，取消下面两行的注释
    # checkpoint = torch.load('your_model_weight.tar', map_location='cpu')
    # model.load_state_dict(checkpoint['state_dict'])

    # 3. 切换到推理模式
    model.eval()

    # 4. 创建伪输入 (Dummy Input)
    # 形状为 [Batch_Size, Channels, Height, Width]
    # SpixelNet 默认输入 3 通道图像。建议尺寸为 16 的倍数（如 160x160）
    dummy_input = torch.randn(1, 3, 160, 160)

    # 5. 定义导出路径
    onnx_file_path = "SpixelNet_Architecture.onnx"

    print(f"正在转换模型至 ONNX...")

    # 6. 执行导出
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,  # 导出模型参数权重
        opset_version=11,  # 算子集版本，v11 兼容性最好
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input_image'],  # 输入节点名称
        output_names=['softmax_mask'],  # 输出节点名称
        # 设置动态维度，这样以后推理时可以输入任意大小的图片
        dynamic_axes={
            'input_image': {0: 'batch_size', 2: 'height', 3: 'width'},
            'softmax_mask': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )

    if os.path.exists(onnx_file_path):
        print(f"🎉 导出成功！文件保存为: {os.path.abspath(onnx_file_path)}")
        print("现在你可以将该文件拖入 https://netron.app/ 查看结构图了。")
    else:
        print("❌ 导出失败，请检查代码。")


if __name__ == "__main__":
    export_model()