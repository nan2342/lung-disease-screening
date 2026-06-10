import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

def create_sample_xray(filename, width=512, height=512):
    """
    创建一个简单的X光样本图像
    """
    try:
        # 创建空白图像
        image = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # 创建胸部X光模拟图像
        # 绘制胸腔轮廓
        for i in range(50):
            # 创建胸腔椭圆
            ellipse_color = (100 + i*2, 100 + i*2, 100 + i*2)
            draw.ellipse((100-i, 100-i, width-100+i, height-100+i), fill=ellipse_color)
        
        # 添加肺部模拟
        # 左肺
        draw.ellipse((150, 150, 350, 350), fill=(180, 180, 180))
        # 右肺
        draw.ellipse((350, 150, width-150, 350), fill=(180, 180, 180))
        
        # 添加心脏区域
        draw.ellipse((230, 250, 380, 400), fill=(150, 150, 150))
        
        # 添加一些纹理
        for _ in range(1000):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            size = np.random.randint(1, 3)
            brightness = np.random.randint(100, 200)
            draw.ellipse((x, y, x+size, y+size), fill=(brightness, brightness, brightness))
        
        # 模拟肺炎的区域
        draw.ellipse((200, 200, 250, 250), fill=(120, 120, 120))
        
        # 保存图像
        image.save(filename)
        print(f"已创建示例X光图像: {filename}")
        return True
    except Exception as e:
        print(f"创建X光图像失败: {e}")
        return False

def create_sample_heatmap(filename, width=512, height=512):
    """
    创建一个简单的热图样本图像
    """
    try:
        # 创建空白图像
        image = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # 绘制与X光图像相似的基本轮廓
        for i in range(50):
            ellipse_color = (30 + i, 30 + i, 30 + i)
            draw.ellipse((100-i, 100-i, width-100+i, height-100+i), fill=ellipse_color)
        
        # 左肺
        draw.ellipse((150, 150, 350, 350), fill=(50, 50, 50))
        # 右肺
        draw.ellipse((350, 150, width-150, 350), fill=(50, 50, 50))
        
        # 心脏区域
        draw.ellipse((230, 250, 380, 400), fill=(40, 40, 40))
        
        # 添加热点区域 - 高风险区域（红色）
        draw.ellipse((200, 200, 250, 250), fill=(255, 50, 50))
        
        # 添加热点区域 - 中等风险区域（黄色）
        draw.ellipse((300, 220, 330, 250), fill=(255, 255, 50))
        
        # 添加热点区域 - 低风险区域（绿色）
        draw.ellipse((350, 280, 370, 300), fill=(50, 255, 50))
        
        # 添加一些纹理
        for _ in range(500):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            size = np.random.randint(1, 3)
            r = np.random.randint(0, 100)
            g = np.random.randint(0, 100)
            b = np.random.randint(0, 100)
            draw.ellipse((x, y, x+size, y+size), fill=(r, g, b))
        
        # 保存图像
        image.save(filename)
        print(f"已创建示例热图图像: {filename}")
        return True
    except Exception as e:
        print(f"创建热图图像失败: {e}")
        return False

def main():
    """
    创建示例图像
    """
    # 创建图像目录
    img_dir = "lung_disease_system/static/img"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    # 示例图像路径
    xray_path = os.path.join(img_dir, "sample-xray.jpg")
    heatmap_path = os.path.join(img_dir, "sample-heatmap.jpg")
    
    # 创建示例图像
    success1 = create_sample_xray(xray_path)
    success2 = create_sample_heatmap(heatmap_path)
    
    if success1 and success2:
        print("所有示例图像创建成功!")
    else:
        print("某些图像创建失败，请检查错误信息。")

if __name__ == "__main__":
    main() 