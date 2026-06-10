import os
import sys
import torch
import torch.nn as nn

# 添加FHRnet路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FHRnet'))
from FHRnet.dcnn import MyDensNet121

def main():

    print("正在将FHRnet中的MyDensNet121模型转换为.pth格式...")
    
    try:
        # 创建模型架构
        model = MyDensNet121(outnum=14, gpsize=4)
        
        # 设置为评估模式
        model.eval()
        
        print("模型架构详情:")
        print(f"- 输入特征通道: 3 (RGB图像)")
        print(f"- 输出类别数: 14 (肺部疾病)")
        print(f"- 模型类型: DenseNet121 (预训练)")
        print(f"- 全局池化大小: 4")
        print(f"- 特征转换: 1024通道卷积层")
        
        # 保存模型权重
        model_path = 'model.pth'
        torch.save(model.state_dict(), model_path)
        print(f"模型权重已保存为 {model_path}")
        
        # 同时保存到lung_disease_system目录
        lung_system_path = os.path.join('lung_disease_system', 'model.pth')
        if not os.path.exists('lung_disease_system'):
            os.makedirs('lung_disease_system')
            
        torch.save(model.state_dict(), lung_system_path)
        print(f"模型权重已复制到 {lung_system_path}")
        
        print("\n注意事项:")
        print("1. 此模型基于预训练的DenseNet121权重")
        print("2. 加载模型时请确保环境中安装了PyTorch")
        print("3. 该模型期望输入224x224的RGB图像")
        print("4. 在Flask应用中，该模型路径已配置好")
        
    except Exception as e:
        print(f"模型创建或保存失败: {e}")
        print("请确保环境中安装了PyTorch和相关依赖项")

if __name__ == "__main__":
    main() 