import time
import sys
import os
sys.path.append(os.pardir)
from UnityEnvClass import Unity_env_deploy

from yolov5.detect_custom1 import *
import argparse
from ros_car import RosCar,LASER_SCAN_SIZE
import time
import cv2
from PIL import Image as PILImage
from stable_baselines3 import SAC, PPO, TD3, REDQ
from stable_baselines3.common.env_util import *
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.tqc.tqc import TQC
from utils.custom_extractors import *
import UnityEnvClass.Unity_env_deploy
from utils.customCallback import TensorboardCallback
from utils.custom_extractors import *

if __name__ == '__main__':
    env = make_unity_env(UnityEnvClass.Unity_env_deploy.UnityBaseEnv, n_envs=1, env_kwargs=dict(file_name=None, time_scale=1,max_steps=2500))
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./yolov5/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
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
    parser.add_argument('--project', default='./yolov5/runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-as-video', action='store_true', help='save same size images as a video')
    parser.add_argument('--submit', action='store_true', help='get submit file in folder submit')
    opt = parser.parse_args()
    print(opt)
    imgsz, device, model, stride, names, colors, half = init(opt)
    car = RosCar()
    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,features_extractor_kwargs=dict(use_resnet=False))
    # REDQ
    # policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,
    #                      features_extractor_kwargs=dict(use_resnet=False),
    #                      n_critics=4
    #                      )
    model = PPO("MultiInputPolicy",env=env,verbose=1,device="cuda",learning_rate=3e-4,seed=42,gamma=0.99,policy_kwargs=policy_kwargs)
    vid_writer = cv2.VideoWriter("origin_ctrl.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (416, 416), True)
    vid_writer2 = cv2.VideoWriter("semantic_ctrl.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (416, 416), True)
    #model = TQC("MultiInputPolicy",env,learning_rate=3e-4,
    #            buffer_size=20000,learning_starts=5000,batch_size=3000,
    #            train_freq=(1,"step"),
    #            top_quantiles_to_drop_per_net=2,
    #            policy_kwargs=policy_kwargs)
    # model = REDQ("MultiInputPolicy", env, learning_rate=3e-4, learning_starts=5000, batch_size=3000,
    #              policy_kwargs=policy_kwargs, buffer_size=20000, train_freq=(1, "step"),
    #              verbose=1, seed=3322
    #              )

    model.set_parameters(load_path_or_dict="./20240320-172136ppo1500.zip")
    while not car.initialized():
        time.sleep(0.5)
    print('car initialized')
    print('laser_scan_size', len(car.laser_scan))
    # print('laser_scan_min_rad', car.laser_scan_min_rad)
    # print('laser_scan_max_rad', car.laser_scan_max_rad)
    car.init_ros_plt()
    while not car.is_shutdown():
        car.update_ros_plt()
        image,ray = car.get_rl_obs_list()
        b, g, r = cv2.split(image)
        img_ = cv2.merge([r, g, b])
        pic = np.uint8(img_ * 255)
        result1, result2, result3 = detect_singlepic(opt, pic, imgsz, device, model, stride, names, colors, half)
        # cv2.imshow('semantic',result2)
        im_pil = PILImage.fromarray(result2)
        im_pil = np.array(im_pil) / 255.0
        cv2.imshow('semantic', im_pil)
        cv2.imshow('origin', img_)
        # cv2.imshow('det',result3)
        # print("det_result:",result1)
        cv2.waitKey(1)
        # print(pic)
        vid_writer.write(pic)
        vid_writer2.write(result2)
        obs = {"image": img_, "semantic":im_pil,"ray": ray}
        action, _states = model.predict(observation=obs, deterministic=True)
        car.move_car(action[0], action[1])

