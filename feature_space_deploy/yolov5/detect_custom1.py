import sys
sys.path.append("./yolov5")
import os
sys.path.append(os.pardir)
import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import torch.nn.functional as F

from models.experimental import attempt_load
from yolo_utils.datasets import LoadStreams, LoadImages
from yolo_utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolo_utils.plots import plot_one_box
from yolo_utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np

Labscene_COLORMAP = [
    [60, 60, 60],# Background
    [255, 0, 0], #S_Obs
    [255, 20, 0],#R_Car
    [0, 128, 0], # W_Car
    [255, 128, 0],#B_Car
    [0, 255, 0],#B_Target
    [128, 0, 128], #Y_obj
    [255, 217, 0], #Wall
    [166, 166, 166],#Plane
    

]

Labscene_IDMAP = [
    [7],
    [8],
    [11],
    [12],
    [13],
    [17],
    [19],
    [20],
    [21]
]

Labscene_Class = ['B_Car','B_Target','R_Car','S_Obs','W_Car','Y_Object','Wall','Plane','_background_']


def label2image(pred, COLORMAP=Labscene_COLORMAP):
    colormap = np.array(COLORMAP, dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]

def trainid2id(pred, IDMAP=Labscene_IDMAP):
    colormap = np.array(IDMAP, dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
def init(args):
    weights, imgsz = args.weights, args.img_size
    set_logging()
    device = select_device(args.device)
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    if half:
        model.half()
    cudnn.benchmark = True
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    print("加载模型完成")
    return  imgsz, device, model, stride, names, colors,half
def detect_singlepic(args,img,imgsz,device,model,stride,names,colors,half):
    
    #print(imgsz)
    t0 = time.time()
    img = letterbox(img, imgsz, stride)[0]
    img0 = img
    #print(img0.shape)
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    
    img = np.ascontiguousarray(img)
    
    #img_o = img
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    #print(img.shape)
    
    # Inference
    with torch.no_grad():
        out = model(img, augment=args.augment)
        pred = out[0][0]
        seg = out[1]  # [0] 1*9*320*416

        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)
    for i, det in enumerate(pred):  # detections per image
        result = []
        #print(len(det))
        #print("i",i)

        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            
            # Write results
            for *xyxy, conf, cls in reversed(det):
                #print(xyxy)
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if args.save_conf else (cls, *xywh)  # label format
                label = f'{names[int(cls)]} {conf:.2f}'
                #print(img0.shape)
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
                result.append(line)


    
        seg = F.interpolate(seg, (416, 416), mode='bilinear', align_corners=True)[0]
        #print(seg.shape)
        mask = label2image(seg.max(axis=0)[1].cpu().numpy(), Labscene_COLORMAP)[:, :, ::-1]

        mask_result = mask
        dst = cv2.addWeighted(mask, 0.4, img0, 0.6, 0)
        print("Done:",time.time()-t0)
        return result,mask_result,img0,dst


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    if opt.submit:
        sub_dir = str(save_dir) + "/results/"
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA 原始代码,cpu用float32,gpu用float16
    # half = False  # 强制禁用float16推理, 20和30系列显卡有tensor cores float16, 10系列卡不开cudnn.benchmark速度反而降
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)
    #print("newsize=",imgsz)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer, s_writer = None, None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
                                # 开启后第一次推理会把各种后端算法测试一遍,后续推理都用最快的算法,会有较明显加速
                                # 算法速度不仅与复杂度有关,也与输入规模相关,因此要求后续输入同尺寸,原版仅在视频测试时开启,想测真实速度应该开启
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        print("webcam!")
    else:
        cudnn.benchmark = False
        dataset = LoadImages(source, img_size=imgsz, stride=stride)  # 跑的是这个
        print("images!")
    if opt.submit or opt.save_as_video:  # 提交和做视频必定是同尺寸
        cudnn.benchmark = True
        
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        #print("image shape is",img.shape) #320*416
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        #print("image shape is",img.shape) 320*416
        # Inference
        with torch.no_grad():
            t1 = time_synchronized()
            #print(img.shape)
            #print(img)
            out = model(img, augment=opt.augment)
            pred = out[0][0]
            seg = out[1]  # [0]
            #print("pred:",pred.shape)
            #print("segresult:",seg.shape)
        # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

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
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                print(det[:,:,4])
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        #print(line[0]==3)
                        #print(line[1])
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        #print("imgshape:",im0)
                        print(xyxy)
                        print(label)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.5f}s)')
            # seg = seg[0]
            #print(im0.shape[0]) 480
            #print(im0.shape[1]) 640
            seg = F.interpolate(seg, (im0.shape[0], im0.shape[1]), mode='bilinear', align_corners=True)[0]
            #print("afterinterpolate:",seg.shape)
            mask = label2image(seg.max(axis=0)[1].cpu().numpy(), Labscene_COLORMAP)[:, :, ::-1]
            dst = cv2.addWeighted(mask, 0.4, im0, 0.6, 0)
            # Stream results
            #print(mask.shape)

            if view_img:
                cv2.imshow("det", im0)
                cv2.imshow("segmentation", mask)
                cv2.imshow("mix", dst)
                #cv2.waitKey(100)&0xff == ord('q')# 1 millisecond
                cv2.waitKey(0)
            if opt.submit:
                sub_path = sub_dir+str(p.name)
                sub_path = sub_path[:-4] + "_pred.png"
                result = trainid2id(seg.max(axis=0)[1].cpu().numpy(), Labscene_IDMAP)
                cv2.imwrite(sub_path, result)
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    #cv2.imwrite(save_path, im0)
                    cv2.imwrite(save_path[:-4]+"_mask"+save_path[-4:], mask)
                    #cv2.imwrite(save_path[:-4]+"_dst"+save_path[-4:], dst)

                else: # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, dst.shape[1], dst.shape[0]
			                #save_path2 = save_path
                            save_path += '.mp4'
                            save_path2 = save_path + 'mask.mp4'
                            save_path3 = save_path + 'origin.mp4'
                       
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer2 = cv2.VideoWriter(save_path2, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer3 = cv2.VideoWriter(save_path3, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

	        
		            
                        #vid_writer2 = cv2.VideoWriter(save_path2, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(dst)#(im0)
                    vid_writer2.write(mask)#(im0)
                    vid_writer3.write(im0)#(im0)


            if opt.save_as_video:
                if not s_writer:
                    fps, w, h = 30, dst.shape[1], dst.shape[0]
                    s_writer = cv2.VideoWriter(str(save_dir)+"out.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                s_writer.write(dst)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    if s_writer != None:
        s_writer.release()
    print(f'Done. ({time.time() - t0:.3f}s)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
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
    parser.add_argument('--save-as-video', action='store_true', help='save same size images as a video')
    parser.add_argument('--submit', action='store_true', help='get submit file in folder submit')
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
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', default=True,action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-as-video', action='store_true', help='save same size images as a video')
    parser.add_argument('--submit', action='store_true', help='get submit file in folder submit')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()'''
