
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import csv
import PIL
import torch
import json
import torch.backends.cudnn as cudnn
from numpy import random
from torchvision import datasets, models, transforms
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
names1 =['鹌鹑蛋','百香果','贝贝南瓜','菠菜','草莓','橙子','豆芽','桂圆','哈密瓜','红圆椒','花菜','黄瓜','黄芒','黄柠檬','黄皮橘子','黄圆椒','胡萝卜','火龙果','鸡蛋','金南瓜','蓝莓','李子','绿甘蓝','猕猴桃','木瓜','牛油果','啤梨','平菇','苹果','葡萄','茄子','芹菜','青尖椒','青萝卜','青芒','青提','青圆椒','秋葵','山楂','山竹','生菜','圣女果','手','水蜜桃','娃娃菜','香蕉','香梨','鲜切哈密瓜','鲜切木瓜','鲜切紫甘蓝','西红柿','西葫芦','西蓝花','雪花梨','樱桃','油菜','油麦菜','油桃','柚子','玉米','紫甘蓝']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1 = torch.load('haha_model_100.pt')
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
'test': transforms.Compose([
        #transforms.Resize(size=256),
        #transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
net = model1.to(device)
def mat_inter(box1, box2):
    # 判断两个矩形是否相交
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False
def solve_coincide(box1, box2):
    # box=(xA,yA,xB,yB)
    # 计算两个矩形框的重合度
    if mat_inter(box1, box2) == True:
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        intersection = col * row
        area1 = (x02 - x01) * (y02 - y01)
        area2 = (x12 - x11) * (y12 - y11)
        coincide = intersection / (area1 + area2 - intersection)
        return coincide
    else:
        return False
f = open('tijiao.csv', 'a', encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(["图片名字", "图片高度", "图片宽度",'类别','候选框'])
def detect(save_img=False):

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    results = []
    boxes=[]
    labels=[]
    label1=[]
    jinzhilist=[]
    newlabels=[]
    newboxes=[]
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:#classify
            pred = apply_classifier(pred, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            boxes = []
            labels = []
            label1 = []
            jinzhilist = []
            newlabels = []
            newboxes = []
            dellist = []
            if len(det):
                #print("有框")

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        boxes.append(xyxy)
                        labels.append(label)
                        label1.append(label.split(' ')[-1])

                        #im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            #print(boxes)
            #print(labels)
            x = np.array(label1)
            y = x.argsort()
            for i in y[::-1]:
                #print("新的一轮"+str(i))
                for j in range(len(y)):
                    if i !=j and i not in jinzhilist:
                        result2 = solve_coincide(boxes[i], boxes[j])
                        if result2>0.1:
                            #print("加进去了")
                            jinzhilist.append(j)
                            #print(j)
            result3=set(jinzhilist)
            #print(result3)
            result4=list(result3)
            for i in range(len(labels)):
                if i not in result4:
                    if float(labels[i].split(' ')[-1])>=0.40:
                        newlabels.append(labels[i])
                        newboxes.append(boxes[i])

            print(newboxes)
            print(newlabels)
            #print(len(newboxes))
            for j in range(len(newboxes)):  # per item
                #print(newboxes[1])
                print("第"+str(j)+"轮循环")
                #print(int(newboxes[j][1]))
                #print(int(newboxes[j][0]))
                #print(int(newboxes[j][2]))
                #print(int(newboxes[j][3]))
                cutout = im0[int(newboxes[j][1]):int(newboxes[j][3]), int(newboxes[j][0]):int(newboxes[j][2])]

                # print(cutout)
                im = cv2.resize(cutout, (224, 224))  # BGR
                cv2.imwrite('test.jpg', im)
                img = PIL.Image.open("test.jpg")
                transform = image_transforms['test']
                img_ = transform(img).unsqueeze(0)
                img_ = img_.to(device)
                outputs = net(img_)
                _, predicted = torch.max(outputs, 1)
                print("预测结果")
                print(names1[int(predicted)])
                print(dellist)
                print(newlabels[j].split(' ')[-1])
                if float(newlabels[j].split(' ')[-1])<=0.85:#0.85
                    print("小于0.83了")
                    if names1[int(predicted)] != '手':
                        newlabels[j] = names1[int(predicted)]
                    print(names1[int(predicted)])
                    if names1[int(predicted)] == '手':
                        #print("加入dellist")
                        dellist.append(j)
            #print(dellist)
            #print(newboxes)
            print("最终的dellist")
            print(dellist)
            if len(dellist):
                for x in dellist:
                    del newboxes[x]
                    del newlabels[x]

                    #del newboxes[j]
                    #del newlabels[j]

                #
            print(newboxes)
            print(newlabels)
            # pred = apply_classifier(newboxes, img, im0s)  # modelc
            for i in range(len(newboxes)):
                plot_one_box(newboxes[i], im0, label=label, color=colors[int(cls)], line_thickness=3)
            for i in range(len(newboxes)):
                labelx = newlabels[i].split(' ')[:-1]
                image_id = str(p).split('\\')[-1]
                boxx = []
                boxx.append(int(newboxes[i][0]))
                boxx.append(int(newboxes[i][1]))
                boxx.append(int(newboxes[i][2]))
                boxx.append(int(newboxes[i][3]))

                # boxx=newboxes[i].detach().numpy()
                csv_writer.writerow([image_id,'720','1280',newlabels[i].split(' ')[0],boxx])
                '''
                result = {
                    'name': image_id,
                    "image_height": 720,
                    "image_width": 1280,
                    "category": newlabels[i].split(' ')[0],
                    "bbox": boxx
                }
                with open('test1.json', 'a+', encoding='utf-8') as file_obj:
                    json.dump(result, file_obj)
                    '''



            #print(results)
            for i in range(len(newboxes)):
                im0 = plot_one_box(newboxes[i], im0, label=newlabels[i], color=colors[int(cls)], line_thickness=3)
            print(f'{s}Done. ({t2 - t1:.3f}s)')


            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp31/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='E:/sysproject/60test/60/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
