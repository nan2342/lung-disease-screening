import os
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import io
import cv2
import datetime
import uuid
import random
import json
import base64
from PIL import Image
from model_utils import ChestXrayModel, DISEASE_CLASSES

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局变量用于存储模型实例
MODEL = None

def load_model():
    """加载并初始化胸部疾病分类模型"""
    global MODEL
    try:
        # 检查模型权重文件是否存在
        model_path = 'static/models/densenet121_chestxray.pth'
        if not os.path.exists(model_path):
            from model_utils import create_sample_model_weights
            model_path = create_sample_model_weights()
            
        # 创建模型实例
        MODEL = ChestXrayModel(model_type='densenet121', model_path=model_path)
        print("胸部疾病分类模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        
def allowed_file(filename):
    """检查文件类型是否允许上传"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/upload')
def upload():
    """上传页面"""
    return render_template('upload.html')

@app.route('/results/<analysis_id>')
def results(analysis_id=None):
    """分析结果页面"""
    if not analysis_id:
        return redirect(url_for('upload'))
        
    # 在实际系统中，应该从数据库获取analysis_id对应的分析结果
    # 这里简化处理，使用当前时间作为分析时间
    analysis_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return render_template(
        'results.html',
        analysis_time=analysis_time,
        analysis_id=analysis_id
    )

@app.route('/about')
def about():
    """关于页面"""
    return render_template('about.html')

@app.route('/download-report')
def download_report():
    """下载报告页面"""
    return render_template('download_report.html')

@app.route('/ai_assistant')
def ai_assistant():
    """AI助手页面"""
    current_time = datetime.datetime.now().strftime("%H:%M")
    return render_template('ai_assistant.html', current_time=current_time)

@app.route('/contact')
def contact():
    """联系我们页面"""
    return "联系我们页面 - 待开发"

@app.route('/community')
def community():
    """社区页面"""
    return render_template('community.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/login') 
def login():
    return render_template('login.html')
@app.route('/health_manager')
def health_manager():
    """健康管理页面"""
    return render_template('health_manager.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理图像预测请求"""
    if 'file' not in request.files:
        return jsonify({'error': '未找到文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件类型'}), 400
    
    try:
        # 生成唯一的分析ID
        analysis_id = f"A{uuid.uuid4().hex[:8]}"
        
        # 保存上传的文件
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{analysis_id}_{filename}")
        file.save(file_path)
        
        # 加载图像
        image = Image.open(file_path).convert('RGB')
        
        # 获取图像尺寸信息
        image_size = f"{image.width} x {image.height}"
        
        # 将原始图像转换为base64编码，以便在前端显示
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        original_image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # 使用模型进行预测
        if MODEL is None:
            load_model()
            
        prediction_results = MODEL.predict(image=image)
        
        # 获取预测结果
        diseases = prediction_results['diseases']  # 前3个概率最高的疾病
        heatmap_base64 = prediction_results['heatmap']  # 热力图
        
        # 构建响应
        response = {
            'success': True,
            'analysis_id': analysis_id,
            'analysis_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'diseases': diseases,
            'image_type': "X射线胸片",  # 可根据实际情况自动识别或让用户选择
            'image_size': image_size,
            'ai_notes': f"系统检测到{diseases[0]['name']}的概率较高({diseases[0]['probability']}%)，请注意关注相关症状。建议进一步临床检查确认诊断。",
            'original_image': f"data:image/jpeg;base64,{original_image_base64}",
            'heatmap_image': f"data:image/jpeg;base64,{heatmap_base64}",
            'redirect_url': f"/results/{analysis_id}"  # 添加重定向URL
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"预测过程出错: {e}")
        return jsonify({'error': f'预测过程出错: {str(e)}'}), 500

if __name__ == '__main__':
    # 启动时加载模型
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)