import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import base64
from FHRnet.dcnn import MyDensNet121, MyResNet50  # 从FHRnet文件夹导入模型类

# 定义14种肺部疾病的类别
DISEASE_CLASSES = {
    0: 'Atelectasis(肺不张)',
    1: 'Cardiomegaly(心脏肥大)',
    2: 'Effusion(胸腔积液)',
    3: 'Infiltration(浸润)',
    4: 'Mass(肿块)',
    5: 'Nodule(结节)',
    6: 'Pneumonia(肺炎)',
    7: 'Pneumothorax(气胸)',
    8: 'Consolidation(实变)',
    9: 'Edema(水肿)',
    10: 'Emphysema(肺气肿)',
    11: 'Fibrosis(纤维化)',
    12: 'Pleural_Thickening(胸膜增厚)',
    13: 'Hernia(疝气)'
}

class ChestXrayModel:
    def __init__(self, model_type='densenet121', model_path=None):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # 创建模型实例
        if model_type == 'densenet121':
            self.model = MyDensNet121(outnum=14)
        elif model_type == 'resnet50':
            self.model = MyResNet50(outnum=14)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 加载预训练权重
        if model_path is not None and os.path.exists(model_path):
            try:
                # 尝试直接加载权重
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except:
                # 如果是checkpoint格式，加载state_dict
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    raise ValueError(f"模型权重格式不正确: {model_path}")
        
        # 设置为评估模式
        self.model.to(self.device)
        self.model.eval()
        
        # 数据预处理转换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path=None, image=None):

        if image_path is not None:
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                raise ValueError(f"无法打开图像文件 {image_path}: {e}")
        
        if image is None:
            raise ValueError("必须提供图像路径或PIL Image对象")
            
        # 应用转换
        tensor = self.transform(image)
        
        # 添加批次维度
        tensor = tensor.unsqueeze(0)
        
        return tensor

    def predict(self, image_path=None, image=None):

        # 预处理图像
        tensor = self.preprocess_image(image_path, image)
        tensor = tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        # 格式化结果
        results = []
        for i, prob in enumerate(probabilities):
            results.append({
                'name': DISEASE_CLASSES[i],
                'probability': round(float(prob * 100), 1)  # 转换为百分比并保留一位小数
            })
        
        # 按概率排序
        results = sorted(results, key=lambda x: x['probability'], reverse=True)
        
        # 生成热力图
        heatmap = self.generate_heatmap(image_path if image_path else image, tensor)
        
        return {
            'diseases': results[:3],  # 只返回前三个最可能的疾病
            'heatmap': heatmap,
            'all_probabilities': probabilities.tolist()
        }
    
    def generate_heatmap(self, image_source, preprocessed_tensor):

        # 加载原始图像
        if isinstance(image_source, str):
            original_image = Image.open(image_source).convert('RGB')
        else:
            original_image = image_source
            
        # 转换为numpy数组并调整大小
        original_image = np.array(original_image.resize((224, 224)))
        
        # 保存原始图像用于混合
        heatmap_image = original_image.copy()
        
        # 获取特征图
        class SaveFeatures:
            def __init__(self, module):
                self.hook = module.register_forward_hook(self.hook_fn)
                self.features = None
            
            def hook_fn(self, module, input, output):
                self.features = output.cpu()
            
            def remove(self):
                self.hook.remove()
        

        if self.model_type == 'densenet121' or self.model_type == 'resnet50':
            target_layer = self.model.features.transit[0]  # 使用第一个卷积层
        else:
            # 默认使用最后一个卷积层
            target_layer = [module for module in self.model.modules() if isinstance(module, nn.Conv2d)][-1]
        
        # 注册钩子
        activations = SaveFeatures(target_layer)
        
        # 再次进行前向传递以获取特征
        with torch.no_grad():
            self.model(preprocessed_tensor.to(self.device))
        
        # 获取特征图并删除钩子
        feature_maps = activations.features[0].numpy()
        activations.remove()
        
        # 计算特征图的平均值，作为热力图
        heatmap = np.mean(feature_maps, axis=0)
        
        # 归一化热力图
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap)
        
        # 使用jet色彩映射
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 将热力图叠加到原始图像上
        superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        
        # 将结果转换为PIL图像
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(superimposed_img)
        
        # 将图像转换为base64编码的字符串
        buffered = io.BytesIO()
        result_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str

# 创建一个临时的模型权重文件
def create_sample_model_weights():
    """创建一个临时的模型权重文件用于测试"""
    model_path = 'static/models/densenet121_chestxray.pth'
    if not os.path.exists(model_path):
        model = MyDensNet121(outnum=14)
        torch.save(model.state_dict(), model_path)
    return model_path

# 测试函数
def test_model():
    model_path = create_sample_model_weights()
    model = ChestXrayModel(model_type='densenet121', model_path=model_path)
    print("模型加载成功!")
    return model 