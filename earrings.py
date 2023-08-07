import dlib
import cv2
import numpy as np
from math import hypot
import math
# 人臉辨識器
detector = dlib.get_frontal_face_detector()
# 人臉特徵器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def rotate(image, angle, center = None, scale = 1):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w/2, h/2)
    # matrix M (旋轉中心, 旋轉角度, 縮放比例)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # 透過 M 進行影像旋轉
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def face_post_glasses(img, Ear, Eyes, Ear2, detector, predictor):
    img = cv2.resize(img, None, fx = 0.8, fy = 0.8)
    (h, w, c) = img.shape
    face = detector(img)
    # 轉成灰階影像
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for face in face:
            landmarks = predictor(img_gray, face)
            for i in range(67):
                x, y = landmarks.part(i).x, landmarks.part(i).y
                # 在特徵點上顯示數字
                # if i == 22 or i == 21 or i == 27 or i == 45 or i == 0 or i == 16:
                    # cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            topl_eyes = (landmarks.part(22).x, landmarks.part(22).y)
            topr_eyes = (landmarks.part(21).x, landmarks.part(21).y)
            center_eyes = (landmarks.part(27).x, landmarks.part(27).y)
            left_eyes = (landmarks.part(45).x, landmarks.part(45).y)
            right_eyes = (landmarks.part(36).x, landmarks.part(36).y)

            eyes_angle = float(-math.atan((right_eyes[1]-left_eyes[1])/(right_eyes[0]-left_eyes[0]))*180/math.pi)
            # Eyes = cv2.cvtColor(Eyes, cv2.COLOR_RGB2BGR)
            Eyes = rotate(Eyes, eyes_angle)

            # 眼睛左與右的位置
            max_left = (int(left_eyes[0]),int(landmarks.part(0).y))
            max_right = (int(right_eyes[0]),int(landmarks.part(16).y))

            eyes_width = int(hypot(left_eyes[0]-right_eyes[0], left_eyes[1]-right_eyes[1])*1.75)
            eyes_height = int(eyes_width)

            # cv2.circle(img, top_left, 5, (255, 0, 0), -1) ####################
            # cv2.circle(img, bottom_right, 5, (255, 0, 0), -1)

            # 改變眼鏡大小，與我眼睛同寬高
            eyes_glass = cv2.resize(Eyes, (eyes_width, eyes_height))

            # 眼鏡變成灰階, 使用閥值變成二值化
            eyes_glass_gray = cv2.cvtColor(eyes_glass, cv2.COLOR_BGR2GRAY)
            _, eyes_mask = cv2.threshold(eyes_glass_gray, 25, 255, cv2.THRESH_BINARY_INV)
            # print('eyes_glass.shape', eyes_glass.shape)
            # print('eyes_mask.shape', eyes_mask.shape)
            # 圖片的角點
            left_x = center_eyes[0] - int(eyes_width/2)
            right_x = center_eyes[0] + int(eyes_width/2)
            height_y = center_eyes[1] - int(eyes_height*4/9)
            bottum_y = center_eyes[1] + int(eyes_height*5/9)
            # 條件式判斷貼圖是否會貼超出畫面
            if (height_y) < 0 or left_x < 0 or bottum_y > h or right_x > w:
                continue
            # 眼鏡預放入的區域大小之眼睛周圍部分
            eyes_area = img[height_y : bottum_y, left_x: right_x]
            # print('eyes_area.shape', eyes_area.shape)
            eyes_area_resized = cv2.resize(eyes_area, (eyes_mask.shape[1], eyes_mask.shape[0]))
            # print('eyes_area_resized.shape', eyes_area_resized.shape)
            # cv2.circle(img, (left_x, height_y), 5, (255, 200, 0), -1) 
            # cv2.circle(img, (right_x, height_y), 5, (255, 0, 0), -1)
            # cv2.circle(img, (left_x, height_y), 5, (255, 100, 0), -1)

            # 每個畫素值進行二進位制“&”操作，1&1=1，1&0=0，0&1=0，0&0=0，
            eyes_area_no_eyes = cv2.bitwise_and(eyes_area_resized, eyes_area_resized, mask=eyes_mask)

            # 將豬鼻子與真鼻子外影像結合的矩形
            final_eyes = cv2.add(eyes_area_no_eyes, eyes_glass)
            final_eyes = cv2.resize(final_eyes, (eyes_area.shape[1], eyes_area.shape[0]))

            # 將矩形放入原來影像之矩形
            img[height_y : bottum_y, left_x: right_x] = final_eyes
##########################
            x, y = landmarks.part(i).x, landmarks.part(i).y
            # 在特徵點上顯示數字
            # if i == 22 or i == 21 or i == 27 or i == 45:
                # cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            left_nose = (landmarks.part(35).x, landmarks.part(35).y)
            right_nose = (landmarks.part(31).x, landmarks.part(31).y)
            center_nose = (landmarks.part(30).x, landmarks.part(30).y)
            left_top = (landmarks.part(14).x, landmarks.part(14).y)
            right_top = (landmarks.part(2).x, landmarks.part(2).y)

            angle = float(-math.atan((right_top[1]-left_top[1])/(right_top[0]-left_top[0]))*180/math.pi)
            # Eyes = cv2.cvtColor(Eyes, cv2.COLOR_RGB2BGR)
            Ear = rotate(Ear, angle)


            eyes_width = int(left_nose[0]-right_nose[0])
            eyes_height = int(eyes_width)

            # 耳垂左與右的位置，即正方形
            top_left = (int(left_top[0]),int(left_top[1]))
            bottom_right = (int(landmarks.part(13).x + eyes_width),int(landmarks.part(13).y))

    #         cv2.circle(img, top_left, 5, (255, 0, 0), -1) ####################
    #         cv2.circle(img, bottom_right, 5, (255, 0, 0), -1)
            # 改變眼鏡大小，與我眼睛同寬高
            eyes_glass = cv2.resize(Ear, (eyes_width, eyes_height))

            # 眼鏡變成灰階, 使用閥值變成二值化
            eyes_glass_gray = cv2.cvtColor(eyes_glass, cv2.COLOR_BGR2GRAY)
            _, eyes_mask = cv2.threshold(eyes_glass_gray, 25, 255, cv2.THRESH_BINARY_INV)
            # print('eyes_glass.shape', eyes_glass.shape)
            # print('eyes_mask.shape', eyes_mask.shape)
            # 圖片的角點
            left_x = left_top[0]
            right_x = left_top[0] + eyes_width
            height_y = center_nose[1] - int(eyes_height/2)
            bottum_y = center_nose[1] + int(eyes_height/2)
            # 條件式判斷貼圖是否會貼超出畫面
            if (height_y) < 0 or left_x < 0 or bottum_y > h or right_x > w:
                continue
            # 眼鏡預放入的區域大小之眼睛周圍部分
            eyes_area = img[height_y : bottum_y, left_x: right_x]
            # print('eyes_area.shape', eyes_area.shape)
            eyes_area_resized = cv2.resize(eyes_area, (eyes_mask.shape[1], eyes_mask.shape[0]))
#                 print('eyes_area_resized.shape', eyes_area_resized.shape)
    #         cv2.circle(img, (left_x, height_y), 5, (255, 200, 0), -1) 
    #         cv2.circle(img, (right_x, height_y), 5, (255, 0, 0), -1)
    #         cv2.circle(img, (left_x, bottum_y), 5, (255, 100, 0), -1)

            # 每個畫素值進行二進位制“&”操作，1&1=1，1&0=0，0&1=0，0&0=0，
            eyes_area_no_eyes = cv2.bitwise_and(eyes_area_resized, eyes_area_resized, mask=eyes_mask)

            # 將豬鼻子與真鼻子外影像結合的矩形
            final_eyes = cv2.add(eyes_area_no_eyes, eyes_glass)
            final_eyes = cv2.resize(final_eyes, (eyes_area.shape[1], eyes_area.shape[0]))

            # 將矩形放入原來影像之矩形
            img[height_y : bottum_y, left_x: right_x] = final_eyes
##########################

            angle = float(-math.atan((right_top[1]-left_top[1])/(right_top[0]-left_top[0]))*180/math.pi)
            # Eyes = cv2.cvtColor(Eyes, cv2.COLOR_RGB2BGR)
            Ear2 = rotate(Ear2, angle)


            eyes_width = int(left_nose[0]-right_nose[0])
            eyes_height = int(eyes_width)

            # 耳垂左與右的位置，即正方形
            top_left = (int(right_top[0]),int(right_top[1]))
            bottom_right = (int(landmarks.part(3).x - eyes_width),int(landmarks.part(3).y))

    #         cv2.circle(img, top_left, 5, (255, 0, 0), -1) ####################
    #         cv2.circle(img, bottom_right, 5, (255, 0, 0), -1)
            # 改變眼鏡大小，與我眼睛同寬高
            eyes_glass = cv2.resize(Ear2, (eyes_width, eyes_height))

            # 眼鏡變成灰階, 使用閥值變成二值化
            eyes_glass_gray = cv2.cvtColor(eyes_glass, cv2.COLOR_BGR2GRAY)
            _, eyes_mask = cv2.threshold(eyes_glass_gray, 25, 255, cv2.THRESH_BINARY_INV)
            # print('eyes_glass.shape', eyes_glass.shape)
            # print('eyes_mask.shape', eyes_mask.shape)
            # 圖片的角點
            left_x = right_top[0] - eyes_width
            right_x = right_top[0] 
            height_y = center_nose[1] - int(eyes_height/2)
            bottum_y = center_nose[1] + int(eyes_height/2)
            # 條件式判斷貼圖是否會貼超出畫面
            if (height_y) < 0 or left_x < 0 or bottum_y > h or right_x > w:
                continue
            # 眼鏡預放入的區域大小之眼睛周圍部分
            eyes_area = img[height_y : bottum_y, left_x: right_x]
            # print('eyes_area.shape', eyes_area.shape)
            eyes_area_resized = cv2.resize(eyes_area, (eyes_mask.shape[1], eyes_mask.shape[0]))
#                 print('eyes_area_resized.shape', eyes_area_resized.shape)
    #         cv2.circle(img, (left_x, height_y), 5, (255, 200, 0), -1) 
    #         cv2.circle(img, (right_x, height_y), 5, (255, 0, 0), -1)
    #         cv2.circle(img, (left_x, bottum_y), 5, (255, 100, 0), -1)

            # 每個畫素值進行二進位制“&”操作，1&1=1，1&0=0，0&1=0，0&0=0，
            eyes_area_no_eyes = cv2.bitwise_and(eyes_area_resized, eyes_area_resized, mask=eyes_mask)

            # 將豬鼻子與真鼻子外影像結合的矩形
            final_eyes = cv2.add(eyes_area_no_eyes, eyes_glass)
            final_eyes = cv2.resize(final_eyes, (eyes_area.shape[1], eyes_area.shape[0]))

            # 將矩形放入原來影像之矩形
            img[height_y : bottum_y, left_x: right_x] = final_eyes

    return img


video_in = cv2.VideoCapture(0)
# video_in = cv2.VideoCapture('Video3.mp4') 放影片
Ear = cv2.imread('images/earringright123.jpg')
Ear = cv2.cvtColor(Ear, cv2.COLOR_BGR2RGB)

Ear2 = cv2.imread('images/earringleft123.jpg')
Ear2 = cv2.cvtColor(Ear2, cv2.COLOR_BGR2RGB)

Eyes = cv2.imread('images/glass1.jpg')
Eyes = cv2.cvtColor(Eyes, cv2.COLOR_BGR2RGB)
while True:
    hasFrame, frame = video_in.read()
    
    img = face_post_glasses(frame, Ear, Eyes, Ear2, detector, predictor)
    
    cv2.imshow('Frame', cv2.flip(img, 1))
#     cv2.imshow('Frame', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_in.release()
cv2.destroyAllWindows()