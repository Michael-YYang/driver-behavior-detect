import keras
import cv2 as cv
import numpy as np
import os

def imgConvert0(img, img_path):
    """闭眼检测图像转换：将RGB图像转换为2值灰度图像，再堆叠为三维图像矩阵，
    以适应网络输入
    参数：
    img-传入图像
    img_path-传入图像的路径，注：二者仅有一个对象被执行
    """
    if img_path:
        img = cv.imread(img_path)
    img0 = cv.resize(img, (24, 24))
    img1 = np.mean(img0, axis=2, keepdims=False)

    # 扩展维数适应预测网络
    img2 = np.expand_dims(img1, axis=3).astype(np.uint8)

    # 增加维度适应网络所需输入
    img3 = np.expand_dims(img2, axis=0)
    
    # 归一化
    my_img = img3 / 255.
    
    return my_img


def imgConvert1(img, img_path):
    """打电话行为检测图片转换：基于皮肤检测算法得到只保留皮肤区域的二值图像"""
    if img_path:
        img = cv.imread(img_path)
    img0 = cv.resize(img, (48, 48))
    # 把图像转换到YUV色域
    ycrcb = cv.cvtColor(img0, cv.COLOR_BGR2YCrCb)  
    
    # 图像分割, 分别获取y, cr, br通道图像
    _, cr, _ = cv.split(ycrcb)  
    
    # 高斯滤波,cr是待滤波的源图像数据,(5,5)是值窗口大小, 0是指根据窗口大小来计算高斯函数标准差
    cr1 = cv.GaussianBlur(cr, (5, 5), 0)
    
    # 根据OTSU算法求图像阈值, 对图像进行二值化
    _, skin1 = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) 
    
    # 扩展维度
    img2 = np.expand_dims(skin1, axis=2)

    # 增加维度适应网络所需输入
    img3 = np.expand_dims(img2, axis=0)
    
    # 归一化
    my_img = img3 / 255.
    
    return my_img


def imgConvert2(img, img_path):
    """抽烟行为检测图像转换，使其可以适应网络输入"""
    if img_path:
        img = cv.imread(img_path)
    img1 = cv.resize(img, (64, 64))
    
    # 增加维度
    img2 = np.expand_dims(img1, axis=0)
    
    # 归一化
    my_img = img2 / 255.
    
    return my_img


def load_model1():
    """载入预训练模型"""
    path = 'model/model_leNet_sleepyDet.hdf5'
    model = keras.models.load_model(path)
    return model


def load_model2():
    """载入预训练模型"""
    path = 'model/model_leNet_phoneDet.hdf5'
    model = keras.models.load_model(path)
    return model


def load_model3():
    """载入MoblieNetv2的模型和权重"""
    # 载入模型
    path = 'model/model_leNet_smokingDet.hdf5'
    model = keras.models.load_model(path)
    return model


def predict_openOrClose(model, img, img_path):
    """预测传入的图像或指定路径的图像"""
    x = imgConvert0(img, img_path)
    result = model.predict(x)

    return result


def predict_isOnPhone(model, img, img_path):
    """预测传入的图像或指定路径的图像"""
    x = imgConvert1(img, img_path)
    result = model.predict(x)
    return result

    
def predict_isSmoking(model, img, img_path):
    """预测传入的图像或指定路径的图像"""
    img = imgConvert2(img, img_path)
    result = model.predict(img)
    
    return result

