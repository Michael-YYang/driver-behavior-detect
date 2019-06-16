import cv2 as cv
from faceDetect import detect_face 
import tensorflow as tf 
import numpy as np
import os
from pre_util import *

def plotBoundingBoxes(image, points, text, color, thickness, lineType=8):
    """根据顶点坐标绘出矩形框"""
    # 检索出检测边界坐标
    x1, y1, x2, y2 = points
    cv.rectangle(image, (x1, y1), (x2, y2), color, thickness, lineType)
    
    # 显示警告内容
    if text:
        cv.putText(image, text, (x1 + 2, y1 - 6), cv.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 255), 1, cv.LINE_AA)


def isSleepyDetect(img, points, Model):
    """根据眼部关键点绘制眼轮廓矩形框，方法：以Mtcnn所获取的眼部关键点为中心，两眼间距为参照，
    选取合适的尺寸分别完成左右眼的跨矩形绘制"""
    # 定义变量表示预测结果，并赋初始值（0-睁眼，1-闭眼）
    flag_left = 0
    flag_right = 0
    
    # 计算两个眼部关键点的距离
    d = np.sqrt(np.square(points[0] - points[1]) + np.square(points[5] - points[6]))\
                                                                .astype(np.int32)
    
    # 获取图像边界
    image_size = np.array(img).shape[0:2]
    
    # 以两个关键点为中心，绘制边界框
    # 第一个边界框坐标
    pointLeft1_x = np.maximum(points[0] - d // 3, 0)  # 左上横坐标
    pointLeft1_y = np.maximum(points[5] - d // 3, 0)  # 左上纵坐标
    pointLeft2_x = points[0] + d // 3  # 右下横坐标
    pointLeft2_y = points[5] + d // 3  # 右下纵坐标
    points1 = (pointLeft1_x, pointLeft1_y, pointLeft2_x, pointLeft2_y)
    
    # 在该区域上裁剪图像
    eye_img1 = img[pointLeft1_y:pointLeft2_y, pointLeft1_x:pointLeft2_x, :]
    
    # 第二个边界框坐标
    pointRight1_x = points[1] - d // 3  # 左上横坐标
    pointRight1_y = points[6] - d // 3  # 左上纵坐标
    pointRight2_x = np.minimum(points[1] + d // 3, image_size[1])  # 右下横坐标         
    pointRight2_y = np.minimum(points[6] + d // 3, image_size[0])  # 右下纵坐标
    points2 = (pointRight1_x, pointRight1_y, pointRight2_x, pointRight2_y)
    
    # 在该区域上裁剪图像
    eye_img2 = img[pointRight1_y:pointRight2_y, pointRight1_x:pointRight2_x, :]

              
    # 在裁剪的人眼局部图像上进行睁-闭眼预测
    result1 = predict_openOrClose(Model, img=eye_img1, img_path=False) 
    result2 = predict_openOrClose(Model, img=eye_img2, img_path=False)
    
    # 绘制眼部轮廓框，预测值大于阈值，绘制绿色框（表示睁眼状态），小于阈值时绘制红色框（表示闭眼状态）
    if result1 < 0.5:
        flag_left = 1
    if result2 < 0.5:
        flag_right = 1
         
    return flag_left, flag_right, points1, points2


def getPhoneDetectRegion(image, size, face_region):
    """定义打电话检测区域，在该区域内裁剪图像，选择性保存，返回边界坐标"""
    # 检索脸部轮廓框坐标
    x1, y1, x2, y2 = face_region
    # 求脸部轮廓框宽度
    d = np.abs(x2 - x1)
    # 打电话检测区域的宽度
    d1 = d // 3
    
    # 定义检测区域的位置（保证在原完整图范围之内）
    # 左边检测框
    X11 = np.maximum(x1 - 2 * d1 , 0)
    Y11 = y2 - int(2.5 * d1)
    X21 = x1 + d1
    Y21 = np.minimum(y2 + int(0.5 * d1), size[0])
    # 顶点坐标
    point_left = (X11, Y11, X21, Y21)
    
    # 右侧检测框
    X12 = x2 - d1
    Y12 = y2 - int(2.5 * d1)
    X22 = np.minimum(x2 + 2 * d1, size[1])
    Y22 = np.minimum(y2 + int(0.5 * d1), size[0])
    # 顶点坐标
    point_right = (X12, Y12, X22, Y22)
    
    # 在检测区域内裁剪图像
    img1 = image[Y11:Y21, X11:X21, :]
    img2 = image[Y12:Y22, X12:X22, :]
    
    return point_left, point_right, img1, img2


def isOnPhoneDetect(img, size, face_region, Model):
    """根据检测到的人脸框对打电话行为检测区域进行定位，对该区域进行检测"""
    # 定义变量表示预测值，并初始化
    flag1 = 0
    flag2 = 0
    
    # 计算检测边界的坐标，并返回在检测区域裁剪的图像
    point1, point2, img1, img2 = getPhoneDetectRegion(img, size, face_region)

    # 载入模型对捕获的检测图像进行预测
    result1 = predict_isOnPhone(Model, img1, img_path=False)
    result2 = predict_isOnPhone(Model, img2, img_path=False)
         
    # 绘制检测框，预测结果大于阈值时在原图上绘制红色框，其余状况绘制绿色框
    if result1 >= 0.5:
        flag1 = 1
    if result2 >= 0.5:
        flag2 = 1
            
    return flag1, flag2, point1, point2


def getSmokingDetectRegion(img, img_size, points):
    """定义吸烟检测区域,裁剪该区域的图像，返回边界坐标"""
    # 检索两个嘴部关键点（嘴角）的位置坐标
    p1, q1, p2, q2 = (points[3], points[8], points[4], points[9])
    
    # 定义检测区域，取两嘴角间距为边长的正方形边界
    # 确定两个顶点的横坐标
    up_left_x = np.maximum(0, p1)
    down_right_x = np.minimum(img_size[1], p2)
    
    # 计算边界框宽度
    d1 = np.abs(down_right_x - up_left_x)    
    
    # 确定两顶点纵坐标
    up_left_y = np.maximum(q1 - int(d1 / 2), 0)
    down_right_y = np.minimum(img_size[0], q1 + int(d1 / 2))
    
    # 得到边界框两个顶点的坐标
    boundingBox = (up_left_x, up_left_y, down_right_x, down_right_y)
    
    # 裁剪检测区域
    det_img = img[up_left_y:down_right_y, up_left_x:down_right_x, :]
    
    return det_img, boundingBox

    
def isSmokingDetect(img, img_size, points,Model):
    """根据检测到的人脸框对打电话行为检测区域进行定位，对该区域进行检测"""
    # 定义变量表示预测值，并初始化
    flag = 0

    # 计算检测边界的坐标，并返回在检测区域裁剪的图像
    det_img, boundingBox = getSmokingDetectRegion(img, img_size, points)

    # 载入模型对捕获的检测图像进行预测
    result1 = predict_isSmoking(Model, det_img, img_path=False)
         
    # 绘制检测框，预测结果大于阈值时在原图上绘制红色框，其余状况绘制绿色框
    if result1 >= 0.5:
        flag = 1
            
    return flag, boundingBox


def driverBehaviorDetect(isCamera, video_path, img_path):
    """
    综合检测疲劳驾驶、打电话、抽烟等不良驾驶行为，以指定时间为检测周期，各行/
    为被检测到的时间占每一检测周期的比值达到阈值则判定为有该不良行为，将给出警告！
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        Model0 = load_model1()
        Model1 = load_model2()
        Model2 = load_model3()
        # 创建P、R、O网络
        pNet, rNet, oNet = detect_face.create_mtcnn(sess, None)
        
        # 定义PRO网络所需参数
        minsize = 20
        threshold = [0.6, 0.7, 0.7]
        facter = 0.709
        margin = 0
        
        # 每轮检测图像帧数
        MAXCOUNT = 20
        # 定义计数器1，视频检测时，记录（每轮）当前图像帧数
        index = 1 
        
        # 定义变量表示每轮闭眼次数
        count_left_eye = 0
        count_right_eye = 0
        
        # 定义打电话行为检测次数记录变量
        count_left_ph = 0
        count_right_ph = 0
        
        # 定义抽烟行为检测次数记录变量
        count_smoking = 0
        
        # 定义计数器2，记录总帧数，可用于存储检测图像时命名
        index1 = 1
        
        if isCamera:
            cap = cv.VideoCapture(0)
        elif video_path:
            cap = cv.VideoCapture(video_path)
        else:
            imageFile = img_path
            if os.path.exists(imageFile):
                image = cv.imread(imageFile, cv.IMREAD_COLOR)
            else:
                print("该路径下图片不存在！")
                
#         if isCamera or video_path:
#             #创建VideoWriter类对象,用以保存检测后的视频
#             fourcc=cv.VideoWriter_fourcc('X','V','I','D')
#             if not os.path.isdir('detectedVideos'):
#                 os.mkdir('detectedVideos')
#             out=cv.VideoWriter('detectedVideos/sleepy.avi',fourcc,20.0,(648, 485))
            
        # 捕获图像
        while True:
            # 读取图像
            if isCamera or video_path:
                ret, image = cap.read()  
                if not ret:
                    continue
                
            # 执行人脸轮廓、关键点检测，返回两个参数表（分别包含多个值）
            image = cv.resize(image, (648, 485))
            boundingBoxes, points = detect_face.detect_face(image, minsize, pNet, \
                                                         rNet, oNet, threshold, facter)
            
            # 获取图像尺寸
            img_size = np.asarray(image.shape)[0:2]
            # 如果检测到多张人脸，获取总人数
            num_face = boundingBoxes.shape[0]
            
            # 若检测到人脸，进一步检测其他对象
            if num_face > 0:
                # 取出第一个结果进行下一步检测
                x, y, x_down, y_down = tuple(np.squeeze(boundingBoxes[0, 0:4]))
                points = points[:, 0].astype(np.int32)  
                
                # 根据前面检测到人脸边界坐标确定缩放后的人脸边界框，margin为扩张尺寸
                x1 = np.maximum(x - margin / 2, 0).astype(np.int32)
                y1 = np.maximum(y - margin / 2, 0).astype(np.int32)
                x_down1 = np.minimum(x_down + margin / 2, img_size[1]).astype(np.int32)
                y_down1 = np.minimum(y_down + margin / 2, img_size[0]).astype(np.int32)
                face_region = (x1, y1, x_down1, y_down1)

                # 打电话行为检测
                flag_phone1, flag_phone2, phone_detect_box1, phone_detect_box2 = \
                                                 isOnPhoneDetect(image,
                                                                img_size,
                                                                face_region,
                                                                Model1)
                
                # 吸烟行为检测
                flag_smoking, smoking_detect_box = isSmokingDetect(image,
                                                                img_size,
                                                                points,
                                                                Model2)
                
                # 检测是否疲劳
                flag_eye1, flag_eye2, eye_detect_box1, eye_detect_box2\
                                               = isSleepyDetect(image,
                                                                points,
                                                                Model0)
                
                # 统计当前轮检测到有打电话行为时间（帧数）
                count_left_ph += flag_phone1
                rate_left_ph = count_left_ph / MAXCOUNT
                count_right_ph += flag_phone2
                rate_right_ph = count_right_ph / MAXCOUNT

                # 统计当前轮检测到抽烟行为的次数
                count_smoking += flag_smoking
                rate_smoking = count_smoking / MAXCOUNT
                
                # 选择性绘制各个行为检测边界
                # 定义变量表示人脸边界、眼部检测边界绘制命令，每帧图像初始化一次
                plot_face = True
                plot_eye = True
                
                # 绘出抽烟行为检测边界
                if flag_smoking:
                    plotBoundingBoxes(image, smoking_detect_box, False, (0, 0, 255), 2)
                    flag_eye1 = 0     # 检测到有抽烟行为时默认没有疲劳行为（防止经常眨眼 引发误判）
                    flag_eye2 = 0  
                    plot_eye = False  # 不用绘制眼部检测边界                      
                
                # 绘出打电话行为检测边界，达到阈值时显示警示语
                if flag_phone1:
                    text1 = 'Using Phone'
                    plotBoundingBoxes(image, phone_detect_box1, text1, (0, 0, 255), 2)
                    plot_face = False  # 检测到有打电话行为时，不再绘制其他检测边界（避免重叠），下同
                    plot_eye = False 
                    flag_eye1 = 0      #检测到有打电话行为时默认没有疲劳行为
                    flag_eye2 = 0   
                if flag_phone2:
                    text2 = 'Using Phone'
                    plotBoundingBoxes(image, phone_detect_box2, text2, (0, 0, 255), 2)
                    plot_face = False
                    plot_eye = False
                    flag_eye1 = 0
                    flag_eye2 = 0     
                        
                # 统计当前轮检测到的闭眼的时间（帧数）
                # 左眼统计
                count_left_eye += flag_eye1
                rate_left_eye = count_left_eye / MAXCOUNT  # 占此轮总时间的比例
                
                # 右眼统计
                count_right_eye += flag_eye2
                rate_right_eye = count_right_eye / MAXCOUNT  # 占此轮总时间的比例
                
                # 绘眼部边界（当检测到闭眼行为，且没有检测到抽烟/打电话行为时绘制）
                if plot_eye:
                    if flag_eye1:
                        plotBoundingBoxes(image, eye_detect_box1, False, (0, 0, 255), 2)
                    if flag_eye2:
                        plotBoundingBoxes(image, eye_detect_box2, False, (0, 0, 255), 2)
                            
                # 绘人脸检测边界（同时负责显示疲劳驾驶、抽烟行为的警示语）
                if plot_face:
                    text = False
                    # 如果有疲劳驾驶行为
                    if plot_eye and  (rate_left_eye > 0.4 or rate_right_eye > 0.4):
                        text = 'Fatigue Driving'
                    # 如果有抽烟的行为
                    if flag_smoking:
                        text = 'Smoking'
                    plotBoundingBoxes(image, (x1, y1, x_down1, y_down1), text, (255, 0, 0), 2)
                
#                 # 画出人脸关键点
#                 cv.circle(image, (points[0], points[5]), 1, (255, 0, 0), thickness=3, lineType=8, shift=0)
#                 cv.circle(image, (points[1], points[6]), 1, (255, 0, 0), thickness=3, lineType=8, shift=0)
#                 cv.circle(image, (points[2], points[7]), 1, (255, 0, 0), thickness=3, lineType=8, shift=0)
#                 cv.circle(image, (points[3], points[8]), 1, (255, 0, 0), thickness=3, lineType=8, shift=0)
#                 cv.circle(image, (points[4], points[9]), 1, (255, 0, 0), thickness=3, lineType=8, shift=0)
            
#             #保存检测后的整幅图
#             if not os.path.isdir('detectedImage/imgDet'):
#                 os.mkdir('detectedImage/imgDet')
#             cv.imwrite('detectedImage/imgDet/e'+str(index)+'.jpg',image)
             
#             #保存检测后的视频
#             if isCamera or video_path:
#                 out.write(image) 
                       
            # 显示图像（视频）
            cv.imshow('driverBehaviorDet', image)
            
            # 设置每帧图像延迟时间 
            if isCamera or video_path:
                if cv.waitKey(40) == 27:
                    break
                
                # 完成一轮检测后，判断是否有不良驾驶行为，若有，抛出警告，并进入下一轮检测
                if index == MAXCOUNT:
                    # 抽烟行为判断
                    if rate_smoking >= 0.5:
                        print("驾车过程中请勿吸烟，注意行车安全！")   
                    
                    # 使用手机行为判断    
                    if rate_left_ph >= 0.5 or rate_right_ph >= 0.5: 
                        print("请遵守交通规则，勿在行车过程中使用手机！")
                    
                    # 疲劳驾驶行为判断
                    if rate_left_eye >= 0.4 and count_right_eye >= 0.4: 
                        print('此轮检测图像帧数：%d,左眼闭眼次数：%d,右眼闭眼次数：%d' % (index,
                                                     count_left_eye, count_right_eye))
                        print("系统检测到你您已处于疲劳状态，请停车休息！")
                         
                    # 完成一轮检测，将计数器置零
                    index = 0
                    count_left_eye = 0
                    count_right_eye = 0
                    count_left_ph = 0
                    count_right_ph = 0
                    count_smoking = 0
                
                # 判断是否检测到人脸（假设只要检测到人脸，就一定完成了其他检测目标），若检测到，计数+1
                if num_face > 0:
                    index += 1
                    index1 += 1
            else:
                if cv.waitKey(0) == 27 :
                    break
    cv.destroyAllWindows()


if __name__ == '__main__':
    """测试三种情况下的检测结果"""
#     #a.接入视频时
#     driverBehaviorDetect(isCamera=True,video_path=False,img_path=False)
 
    # b.输入视频路径
    path = 'E:/pydev/challenge/originData/sleepy.avi'
    driverBehaviorDetect(isCamera=False, video_path=path, img_path=False)
     
#     # c.输入为图像路径时
#     img_path = 'E:/pydev/challenge/originData/image/img1.jpg'
#     driverBehaviorDetect(isCamera=False, video_path=False, img_path=img_path)

'''
Created on 2019年6月3日

@author: Administrator
'''
