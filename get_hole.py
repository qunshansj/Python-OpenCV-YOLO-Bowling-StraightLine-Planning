
import torch, cv2, os, tqdm
import numpy as np

import numpy as np

from yolov5.yolo import YOLOV5_Detect, opt

def get_hole(pred):
    pred[pred == 1] = 0
    pred[pred != 0] = 255
    contours, hierarchy = cv2.findContours(pred[:, :, 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def distance(x1,y1,x2,y2):
    return ((x1 - x2)**2+(y1 - y2)**2)**0.5

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    model = torch.load('model.pkl')
    model.to(DEVICE)

    if not os.path.exists('output'):
        os.mkdir('output')

    path = 'detect'


    yolo_detect = YOLOV5_Detect(**vars(opt))

    for i in tqdm.tqdm(os.listdir(path)):
        # 读取图像
        ori_img = cv2.imdecode(np.fromfile('{}/{}'.format(path, i), np.uint8), cv2.IMREAD_COLOR)
        yolov5_res,balllist,ballvalue = yolo_detect.detect(ori_img)
        # 记录原图尺寸
        img_shape = ori_img.shape
        # Resize到训练大小 640*320
        img_ = cv2.resize(ori_img, (640, 320))
        # 转换通道 归一化
        img = np.transpose(np.expand_dims(img_, axis=0), (0, 3, 1, 2)) / 255.0
        # 转换成tensor格式
        img = torch.from_numpy(img).to(DEVICE).float()
        # 预测
        pred = np.argmax(model(img).cpu().detach().numpy()[0], axis=0)

        # 1 2 对应着目标类别
        pred_mask = []
        for j in pred.reshape((-1)):
            if j == 0:
                pred_mask.append(np.array([0, 0, 0]))
            elif j == 1:
                pred_mask.append(np.array([0, 0, 255]))
            elif j == 2:
                pred_mask.append(np.array([255, 0, 0]))
        pred_mask = np.array(pred_mask, dtype=np.uint8).reshape((pred.shape[0], pred.shape[1], 3))

        pred = np.expand_dims(pred, axis=-1)
        pred = np.repeat(pred, axis=-1, repeats=3)
        pred = np.array(pred, dtype=np.uint8)
        pred = cv2.resize(pred, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
        pred_mask = cv2.resize(pred_mask, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
        contours = get_hole(pred)

        pred[pred == 0] = 255
        pred[pred != 255] = 0
        yolov5_res = yolov5_res & pred
        # yolov5_res = cv2.addWeighted(yolov5_res, 0.5, pred_mask, 0.5, 0)
        # ori_img = yolov5_res & pred

        holes_coordinate = []
        radius = 0
        number = 0
        for cnts in contours:
            x, y, w, h = cv2.boundingRect(cnts)
            holes_coordinate.append([x, y, w, h])
            radius = radius + (w + h)/2
            number = number + 1
        holllist = []
        hollvalue = []
        holes_coordinate = sorted(holes_coordinate, key=lambda x:x[2] * x[3], reverse=True)[:6]
        holes_coordinate = sorted(holes_coordinate, key=lambda x:x[1])
        for idx, (x, y, w, h) in enumerate(sorted(holes_coordinate[:3], key=lambda x:x[0])):
            cv2.rectangle(yolov5_res, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(yolov5_res, '{:.0f}'.format(idx + 1),
                        (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 0, 0), 2)
            holllist.append(idx + 1)
            hollvalue.append([x + w / 2, y + h / 2, (w + h) / 2])
            #print('hole {} x_center:{:.2f} y_center:{:.2f} radius:{:.2f}'.format(idx + 1, x + w / 2, y + h / 2, (w + h) / 2))

        for idx, (x, y, w, h) in enumerate(sorted(holes_coordinate[3:], key=lambda x:x[0])):
            cv2.rectangle(yolov5_res, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(yolov5_res, '{:.0f}'.format(idx + 4),
                        (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 0, 0), 2)
            holllist.append(idx + 4)
            hollvalue.append([x + w / 2, y + h / 2, (w + h) / 2])
            #print('hole {} x_center:{:.2f} y_center:{:.2f} radius:{:.2f}'.format(idx + 4, x + w / 2, y + h / 2, (w + h) / 2))
        for m in range(len(balllist)):
            cv2.circle(yolov5_res, (int(ballvalue[m][0]),int(ballvalue[m][1])), int(ballvalue[m][2]), [0,0,255], 2)
        cv2.imshow('input', yolov5_res)
        cv2.waitKey(0)
        print(balllist)
        print(ballvalue)
        print(holllist)
        print(hollvalue)
        a = input("请输入母球编号: ")
        b = input("请输入目标球编号: ")
        c = input("请输入袋口编号: ")
        x1 = int(ballvalue[balllist.index(int(a))][0])
        y1 = int(ballvalue[balllist.index(int(a))][1])
        r1 = int(ballvalue[balllist.index(int(a))][2])
        x2 = int(ballvalue[balllist.index(int(b))][0])
        y2 = int(ballvalue[balllist.index(int(b))][1])
        r2 = int(ballvalue[balllist.index(int(b))][2])
        x3 = int(hollvalue[holllist.index(int(c))][0])
        y3 = int(hollvalue[holllist.index(int(c))][1])
        r3 = int(hollvalue[holllist.index(int(c))][2])
        #画出目标球的可能行进路线：
        def drawline(yolov5_res,x1,y1,x2,y2,r):
            gen = ((x2-x1)**2+(y2-y1)**2)**0.5
            x3 = int(x1 - (y2-y1)*r/gen)
            y3 = int(y1 + (x2-x1)*r/gen)
            x4 = int(x2 - (y2-y1)*r/gen)
            y4 = int(y2 + (x2-x1)*r/gen)
            x5 = int(x1 + (y2 - y1) * r / gen)
            y5 = int(y1 - (x2 - x1) * r / gen)
            x6 = int(x2 + (y2 - y1) * r / gen)
            y6 = int(y2 - (x2 - x1) * r / gen)
            cv2.line(yolov5_res, (x3, y3), (x4, y4), (255, 255, 255), 3)
            cv2.line(yolov5_res, (x5, y5), (x6, y6), (255, 255, 255), 3)
            return yolov5_res

        cv2.line(yolov5_res,(x2,y2),(x3,y3),(255, 0, 0),3)
        yolov5_res = drawline(yolov5_res,x2,y2,x3,y3,r2)
        #画出撞击点
        xz = int(x3 + (x2 - x3)*((r1+r2) + distance(x2,y2,x3,y3))/(distance(x2,y2,x3,y3)))
        yz = int(y3 + (y2 - y3)*((r1+r2) + distance(x2,y2,x3,y3))/(distance(x2,y2,x3,y3)))
        cv2.circle(yolov5_res, (xz,yz), int(ballvalue[balllist.index(int(a))][2]), [255, 0, 0], 2)
        #画出母球的行进路线
        cv2.line(yolov5_res, (xz, yz), (x1, y1), (255, 0, 0), 3)
        yolov5_res = drawline(yolov5_res, xz, yz, x1, y1,r1)
        cv2.imshow('output',yolov5_res)
        cv2.waitKey(0)

        #cv2.imwrite('output/{}'.format(i), yolov5_res)
