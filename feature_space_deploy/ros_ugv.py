"""
ROS小车底层驱动
"""

import math
import os
import struct
import threading
import time

import cv2
import cv_bridge
import matplotlib.pyplot as plt
import numpy as np
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from limo_base.msg import LimoStatus
from nav_msgs.msg import Odometry
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import CompressedImage, Image, LaserScan

DEPTH_MAX = 3.  # 米

LASER_SCAN_SIZE = 400
LASER_SCAN_MAX = 3.  # 米
_ros2rl_laser_index = []  # ROS -> RL 雷达转换数组
for i in range(LASER_SCAN_SIZE // 2, 0, -1):
    _ros2rl_laser_index.append(i - 1)
    _ros2rl_laser_index.append(LASER_SCAN_SIZE - i)

_ros2rl_convert_mat = np.array([[[0, 1], [-1, 0]]])  # ROS -> RL 坐标转换矩阵

RL_CAMERA = 'CAMERA'
RL_DEPTH = 'DEPTH'
RL_LASER = 'LASER'
RL_STATE = 'STATE'
RL_ABS_STATE = 'ABS_STATE'


def resize_camera(im_np):
    h, w = im_np.shape[:2]
    im_np = im_np[:, (w - h) // 2: (w + h) // 2]
    im_np = cv2.resize(im_np, (84, 84))

    return im_np


def quart_to_rpy(x, y, z, w):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw


def ros_xy_to_plt(x, y):
    return y, x


class RosUGV:
    _closed = False

    _bridge = cv_bridge.CvBridge()

    camera_height = None  # 图像高度 (ROS格式)
    camera_width = None  # 图像宽度 (ROS格式)
    camera_image = None  # 缓存的图像 (ROS格式)
    camera_time = None  # 最后一帧图像获取时间

    depth_camera_height = None  # 深度图像高度 (ROS格式)
    depth_camera_width = None  # 深度图像宽度 (ROS格式)
    depth_camera_image = None  # 缓存的深度图像 (ROS格式)
    depth_camera_time = None  # 最后一帧深度图像获取时间

    laser_scan = None  # 缓存的雷达数据 (ROS格式)
    laser_scan_intensity = None  # 缓存的雷达强度数据 (ROS格式)
    laser_scan_time = None  # 最后一帧雷达获取时间

    position = init_position = None  # 缓存的位置坐标、初始位置坐标 (ROS格式，根据自身里程计获得，不精确)
    yaw = init_yaw = None  # 缓存的角度、初始角度 (ROS格式，根据自身imu获得，较精确)
    velocity = None  # 缓存的速度向量 (ROS格式，根据自身imu获得，较精确)
    pose_time = None  # 最后一帧小车状态获取时间

    abs_position = None  # 缓存的绝对位置坐标 (ROS格式，根据 UWB 获得)
    abs_position_time = None  # 最后一帧绝对位置坐标获取时间

    control_mode = None  # 小车控制模式 (ROS格式)
    motion_mode = None  # 小车运动模式 (ROS格式)

    ema = 0.2
    pre_average_motor = 0.
    pre_average_steer = 0.

    def __init__(self, ROS_MASTER_URI=None, ROS_IP=None,
                 motor_limit=0.25,
                 ema=0.2,
                 repeat_action=False,
                 active_rl_keys_list=None,
                 raw_laser_for_rl=False,
                 gray_camera=False):
        """
        Args:
            ROS_MASTER_URI: ROS MASTER 地址, 若为None, 则自动读取环境变量
            ROS_IP: ROS IP 地址, 若为None, 则自动读取环境变量
            motor_limit: 最大油门限制
            ema: 移动平均比率
            repeat_action: 在没有动作指令的情况下, 是否一直重复上一时刻的动作（10Hz）
            active_rl_keys_list: 需要使用的传感器种类
            raw_laser_for_rl: 获得的强化学习观测值中的雷达射线是否为原始的雷达感知数据。是的话雷达感知返回为400维,否则为802维
            gray_camera: 摄像头是否输出灰度图（默认为彩色图）
        """
        if ROS_MASTER_URI is not None:
            os.environ['ROS_MASTER_URI'] = ROS_MASTER_URI
        if ROS_IP is not None:
            os.environ['ROS_IP'] = ROS_IP

        print('ROS_MASTER_URI', ROS_MASTER_URI)
        print('ROS_IP', ROS_IP)
        print('ROS_NAMESPACE', os.environ['ROS_NAMESPACE'])

        self.motor_limit = motor_limit
        self.ema = ema
        self.repeat_action = repeat_action
        self.raw_laser_for_rl = raw_laser_for_rl
        self.gray_camera = gray_camera

        if active_rl_keys_list is None:
            self.active_rl_keys_list = [RL_CAMERA, RL_DEPTH, RL_LASER, RL_STATE, RL_ABS_STATE]
        else:
            self.active_rl_keys_list = active_rl_keys_list

        # 创建一个ROSPublisher，用于发布控制小车运动的指令
        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

        # 初始化 ROS 节点
        rospy.init_node('ros_server_listener', anonymous=True, disable_signals=True)

        # 创建相应的订阅器来订阅 ROS Topics，以获取传感器数据或小车状态信息。
        if RL_CAMERA in self.active_rl_keys_list:
            rospy.Subscriber('camera/image_raw/compressed/resized',
                             Image, self._camera_callback)
        if RL_DEPTH in self.active_rl_keys_list:
            rospy.Subscriber('camera/depth/image_raw',  # compressedDepth 帧率极低
                             Image, self._depth_camera_callback)
        if RL_LASER in self.active_rl_keys_list:
            rospy.Subscriber('scan', numpy_msg(LaserScan), self._scan_callback)

        rospy.Subscriber('odom', numpy_msg(Odometry), self._odom_callback)  # 小车自身信息，永远获取
        rospy.Subscriber('limo_status', LimoStatus, self._limo_status_callback)  # 小车底盘信息，永远获取

        if RL_ABS_STATE in self.active_rl_keys_list:
            rospy.Subscriber('position', Twist, self._position_callback)

        # 如果重复指令，则开启无限发送动作指令的线程
        if self.repeat_action:
            self.cmd = None
            t = threading.Thread(target=self._repeat_send_cmd_vel)
            t.start()

        t = threading.Thread(target=self._check_data_time)
        t.start()

    def display_status(self):
        print(f"""===初始化检查开始===    当前值 \t目标值
camera_image:           {self.camera_time is not None}  \t{RL_CAMERA in self.active_rl_keys_list}
depth_camera_image:     {self.depth_camera_time is not None}  \t{RL_DEPTH in self.active_rl_keys_list}
laser_scan:             {self.laser_scan_time is not None}  \t{RL_LASER in self.active_rl_keys_list}
pose:                   {self.pose_time is not None}  \t{RL_STATE in self.active_rl_keys_list}
abs_position:           {self.abs_position is not None}  \t{RL_ABS_STATE in self.active_rl_keys_list}
===初始化检查结束===""")

    def initialized(self):
        """
        小车是否已经初始化完毕（传感信息是否已经能够接收），建议循环等待
        """
        return all([self.camera_time is not None if RL_CAMERA in self.active_rl_keys_list else True,
                    self.depth_camera_time is not None if RL_DEPTH in self.active_rl_keys_list else True,
                    self.laser_scan_time is not None if RL_LASER in self.active_rl_keys_list else True,
                    self.pose_time is not None,
                    self.abs_position_time is not None if RL_ABS_STATE in self.active_rl_keys_list else True])

    def _check_data_time(self):
        """
        检查获取到的传感数据是否中断超时
        由于底盘问题，小车状态信息可能会中断，建议断开电源，重启小车底盘
        """
        while not self._closed:
            if not self.initialized():
                continue

            if self.camera_time is not None and time.time() - self.camera_time > 1:
                print('camera image 超时')
            if self.depth_camera_time is not None and time.time() - self.depth_camera_time > 1:
                print('depth camera image 超时')
            if self.laser_scan_time is not None and time.time() - self.laser_scan_time > 1:
                print('laser 超时')
            if self.pose_time is not None and time.time() - self.pose_time > 1:
                print('pose 超时')
            if self.abs_position_time is not None and time.time() - self.abs_position_time > 1:
                print('abs_pose 超时')
            time.sleep(1)

    # def _camera_callback(self, data: CompressedImage):
    #     """
    #     摄像机传感数据回调函数
    #     """
    #     np_arr = np.frombuffer(data.data, np.uint8)
    #     im_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    #     if self.gray_camera:
    #         im_np = cv2.cvtColor(im_np, cv2.COLOR_BGR2GRAY)
    #     else:
    #         im_np = cv2.cvtColor(im_np, cv2.COLOR_BGR2RGB)

    #     self.camera_height = im_np.shape[0]
    #     self.camera_width = im_np.shape[1]
    #     self.camera_image = im_np

    #     self.camera_time = data.header.stamp.to_time()

    def _camera_callback(self, data: Image):
        """
        摄像机传感数据回调函数
        """
        cv_image = self._bridge.imgmsg_to_cv2(data)
        if self.gray_camera:
            im_np = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        else:
            im_np = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        self.camera_height = im_np.shape[0]
        self.camera_width = im_np.shape[1]
        self.camera_image = im_np

        self.camera_time = data.header.stamp.to_time()

    def _depth_camera_callback(self, data: Image):
        """
        深度摄像机传感数据回调函数
        """
        cv_image = self._bridge.imgmsg_to_cv2(data)  # uint16
        im_np = np.array(cv_image, dtype=np.uint16)
        im_np[im_np == 0] = DEPTH_MAX * 1000  # 失效的像素点置为最远距离
        im_np = im_np.clip(0, DEPTH_MAX * 1000).astype(np.float32) / (DEPTH_MAX * 1000) * 255.
        im_np = im_np.astype(np.uint8)

        # # CompressedDepth (帧率太低，暂时取消)
        # depth_header_size = 12
        # raw_data = data.data[depth_header_size:]
        # depth_img_raw = cv2.imdecode(np.frombuffer(raw_data, np.uint8), cv2.IMREAD_UNCHANGED)
        # raw_header = data.data[:12]
        # # header: int, float, float
        # [compfmt, depthQuantA, depthQuantB] = struct.unpack('iff', raw_header)
        # depth_img_scaled = depthQuantA / (depth_img_raw.astype(np.float32) - depthQuantB)
        # # filter max values
        # depth_img_scaled[depth_img_raw == 0] = 0.

        # im_np = np.clip(depth_img_scaled, 0., DEPTH_MAX)

        # im_np = im_np / DEPTH_MAX

        self.depth_camera_height = im_np.shape[0]
        self.depth_camera_width = im_np.shape[1]
        self.depth_camera_image = im_np

        self.depth_camera_time = data.header.stamp.to_time()

    def get_camera_image(self):
        """
        获取摄像机传感数据 (ROS格式)
        大小为 (self.camera_height, self.camera_width, 3), dtype=np.uint8
        """
        return self.camera_image

    def get_depth_camera_image(self):
        """
        获取深度摄像机传感数据 (ROS格式)
        大小为 (self.depth_camera_height, self.depth_camera_width), dtype=np.float32
        """
        return self.depth_camera_image

    def _scan_callback(self, data: LaserScan):
        """
        雷达传感数据回调函数
        """
        self.laser_scan_min_rad = data.angle_min
        self.laser_scan_max_rad = data.angle_max
        self.laser_scan = data.ranges
        self.laser_scan_intensity = data.intensities

        self.laser_scan_time = data.header.stamp.to_time()

    def get_laser_scan(self):
        """
        获取雷达传感数据 (ROS格式)
        一维向量, 长度不定, 在400左右
        """
        return self.laser_scan

    def _odom_callback(self, data: Odometry):
        """
        小车自身里程计信息回调函数
        """
        position = data.pose.pose.position
        self.position = np.array([position.x, position.y])

        orientation = data.pose.pose.orientation
        roll, pitch, yaw = quart_to_rpy(orientation.x,
                                        orientation.y,
                                        orientation.z,
                                        orientation.w)
        self.yaw = yaw

        if self.init_position is None or self.init_yaw is None:
            self.reset()

        linear = data.twist.twist.linear
        self.velocity = np.array([linear.x, linear.y])

        self.pose_time = data.header.stamp.to_time()

    def get_pose(self):
        """
        获取小车自身信息 (ROS格式)
        [速度x, 速度y, 角度, 位置x, 位置y]
        """
        return np.concatenate([self.velocity, np.array([self.yaw]), self.position])

    def _limo_status_callback(self, data: LimoStatus):
        """
        limo小车独有状态信息回调函数, 获取小车运动模式
        """
        if data.control_mode != self.control_mode:
            print('小车控制状态更新:')
            self.control_mode = data.control_mode
            if data.control_mode == 1:
                print('ROS控制')
            elif data.control_mode == 2:
                print('手柄控制')
            else:
                print('CONTROL MODE UNKNOWN', data.motion_mode)
        if data.motion_mode != self.motion_mode:
            print('小车运动状态更新:')
            self.motion_mode = data.motion_mode
            if data.motion_mode == 0:
                print('四轮差速')
            elif data.motion_mode == 1:
                print('阿克曼')
            elif data.motion_mode == 2:
                print('麦轮')
            else:
                print('MODE_UNKNOWN')

    def _position_callback(self, data: Twist):
        self.abs_position = np.array([data.linear.x, data.linear.y, data.linear.z])
        self.abs_position_time = time.time()

    def _repeat_send_cmd_vel(self):
        """
        发送运动指令线程, 每秒10帧数据
        """
        r = rospy.Rate(10)
        while not self._closed:
            if self.cmd is not None:
                self.cmd_pub.publish(self.cmd)

            r.sleep()

    def _get_processed_laser_scan(self):
        """
        获取处理后的雷达数据, 统一到400条射线 (ROS格式)
        """
        laser_scan = self.laser_scan
        laser_scan_intensity = self.laser_scan_intensity
        laser_scan_size = len(laser_scan)
        size_delta = LASER_SCAN_SIZE - laser_scan_size
        if size_delta > 0:  # 如果真实的射线数较少
            idx = [0] * math.floor(size_delta / 2) + [laser_scan_size] * math.ceil(size_delta / 2)
            laser_scan = np.insert(laser_scan, idx, 0.03)
            laser_scan_intensity = np.insert(laser_scan_intensity, idx, 0.)
        elif size_delta < 0:  # 如果真实的射线数较多
            idx = np.random.choice(laser_scan_size, -size_delta, replace=False)
            laser_scan = np.delete(laser_scan, idx)
            laser_scan_intensity = np.delete(laser_scan_intensity, idx)

        return laser_scan, laser_scan_intensity

    def move_ugv(self, x, angle):
        """
        发送移动小车的动作指令 (ROS指令)
        """
        cmd = Twist()  # 创建了一个空的Twist类型的消息对象叫cmd
        cmd.linear.x = x
        cmd.angular.z = angle
        # print(f'x: {x:2f}, az: {angle:2f}')

        if self.repeat_action:
            self.cmd = cmd
        else:
            self.cmd_pub.publish(cmd)

    def send_rl_action(self, action):
        """
        发送移动小车的强化学习动作指令 (RL指令)
        [方向, 油门]
        方向 in [-1, 1], -1表示向左, 1表示向右
        油门 in [-1, 1], -1表示向后, 1表示向前
        """
        action = action[0]
        steer, motor = action

        motor = (1 - self.ema) * motor + self.ema * self.pre_average_motor
        steer = (1 - self.ema) * steer + self.ema * self.pre_average_steer
        self.pre_average_motor = motor
        self.pre_average_steer = steer

        # motor_candi = np.linspace(-1, 1, 5)
        # steer_candi = np.linspace(-1, 1, 5)

        # motor = motor_candi[np.abs(motor_candi - motor).argmin()]
        # steer = steer_candi[np.abs(steer_candi - steer).argmin()]

        if motor >= 0:
            steer = -steer

        self.move_ugv(motor * self.motor_limit, steer)

    def get_rl_obs_shapes(self):
        """
        获取强化学习适用的观测维度 (RL格式)
        (摄像头, 深度摄像头, 雷达, 小车自身信息)
        """
        obs_shapes = {
            RL_CAMERA: (84, 84, 3) if not self.gray_camera else (84, 84, 1),
            RL_DEPTH: (84, 84, 1),
            RL_LASER: ((LASER_SCAN_SIZE + 1) * 2, ) if not self.raw_laser_for_rl else (LASER_SCAN_SIZE, ),
            RL_STATE: (6,),
            RL_ABS_STATE: (6,)
        }
        return [obs_shapes[k] for k in self.active_rl_keys_list]

    def get_rl_action_size(self):
        """
        获取强化学习适用的动作维度 (RL格式)
        """
        return 2

    def get_rl_obs_list(self):
        """
        获取强化学习适用的观测值 (RL格式)
        (摄像头数据, 深度摄像头数据, 雷达数据, 小车自身信息数据)
        雷达为unity格式, 没有探测到的射线一律设置(1,1)
        小车自身信息为 (速度x, 速度z, 方向向量x, 方向向量z, 位置x, 位置z)
            方向向量与位置均为相对初始方向向量与初始位置的相对量，可通过 `reset` 重置初始状态
        所有维度与 `get_rl_obs_shapes` 一致
        """
        # CAMERA
        rl_camera_obs = None
        if self.camera_time is not None:
            ros_camera = self.camera_image
            rl_camera_obs = np.expand_dims(resize_camera(ros_camera) / 255., 0).astype(np.float32)
            if self.gray_camera:
                rl_camera_obs = np.expand_dims(rl_camera_obs, -1)

        # DEPTH_CAMERA
        rl_depth_camera_obs = None
        if self.depth_camera_time is not None:
            ros_depth_camera = self.depth_camera_image
            rl_depth_camera_obs = np.expand_dims(resize_camera(ros_depth_camera) / 255., -1)
            rl_depth_camera_obs = np.expand_dims(rl_depth_camera_obs, 0).astype(np.float32)
            rl_depth_camera_obs = (rl_depth_camera_obs + 0.3).clip(0., 1.).astype(np.float32)

        # LASER
        rl_laser_scan_obs = None
        if self.laser_scan_time is not None:
            ros_laser_scan, ros_laser_scan_intensity = self._get_processed_laser_scan()
            if self.raw_laser_for_rl:
                rl_laser_scan = [min(d / LASER_SCAN_MAX, 1.) if i > 1000 else 1. for d, i in zip(ros_laser_scan, ros_laser_scan_intensity)]
                rl_laser_scan_obs = np.array([rl_laser_scan], dtype=np.float32)
            else:
                rl_laser_scan = ros_laser_scan[_ros2rl_laser_index]
                rl_laser_scan_intensity = ros_laser_scan_intensity[_ros2rl_laser_index]

                rl_laser_scan_hashit = np.ones_like(rl_laser_scan)
                rl_laser_scan_hashit[np.logical_and(rl_laser_scan < LASER_SCAN_MAX, rl_laser_scan_intensity > 1000)] = 0.

                rl_laser_scan_hitfraction = np.ones_like(rl_laser_scan)
                _mask = rl_laser_scan_intensity > 1000
                rl_laser_scan_hitfraction[_mask] = np.minimum(rl_laser_scan[_mask] / LASER_SCAN_MAX, 1.)

                rl_laser_scan = np.stack([rl_laser_scan_hashit, rl_laser_scan_hitfraction], axis=-1)  # [LASER_SCAN_SIZE, 2]
                rl_laser_scan = rl_laser_scan.flatten()  # [LASER_SCAN_SIZE * 2, ]
                rl_laser_scan = np.concatenate([np.zeros(2), rl_laser_scan]).astype(np.float32)  # [ 2 + LASER_SCAN_SIZE * 2, ]
                rl_laser_scan_obs = np.expand_dims(rl_laser_scan, 0)  # [1, 2 + LASER_SCAN_SIZE * 2]

        # POSE
        ros_velocity = self.velocity
        rl_velocity = np.matmul(ros_velocity, _ros2rl_convert_mat).squeeze(0)

        # RELATIVE POSE
        rl_pose_obs = None
        if self.pose_time is not None:
            ros_yaw = self.yaw - self.init_yaw
            ros_position = self.position - self.init_position
            rotate_matrix = np.array([[math.cos(self.init_yaw), -math.sin(self.init_yaw)],
                                      [math.sin(self.init_yaw), math.cos(self.init_yaw)]])
            ros_position = np.dot(ros_position, rotate_matrix)
            rl_yaw = -ros_yaw
            rl_forward = np.array([math.sin(rl_yaw), math.cos(rl_yaw)])
            rl_position = np.matmul(ros_position, _ros2rl_convert_mat).squeeze(0)
            rl_pose = np.concatenate([rl_velocity, rl_forward, rl_position]).astype(np.float32)
            rl_pose_obs = np.expand_dims(rl_pose, 0)

        # ABS POSE
        rl_abs_pose_obs = None
        if self.abs_position_time is not None:
            ros_yaw = self.yaw
            ros_position = self.abs_position[:2]
            rl_yaw = -ros_yaw
            rl_forward = np.array([math.sin(rl_yaw), math.cos(rl_yaw)])
            rl_position = np.matmul(ros_position, _ros2rl_convert_mat).squeeze(0)
            rl_abs_pose = np.concatenate([rl_velocity, rl_forward, rl_position]).astype(np.float32)
            rl_abs_pose_obs = np.expand_dims(rl_abs_pose, 0)

        obs_list = {
            RL_CAMERA: rl_camera_obs,
            RL_DEPTH: rl_depth_camera_obs,
            RL_LASER: rl_laser_scan_obs,
            RL_STATE: rl_pose_obs,
            RL_ABS_STATE: rl_abs_pose_obs
        }

        return [obs_list[k] for k in self.active_rl_keys_list]

    def init_ros_plt(self):
        """
        初始化ROS原始传感数据的可视化界面
        """
        self._ros_fig = plt.figure()
        grid = plt.GridSpec(nrows=4, ncols=4, wspace=0.2, hspace=0.2)
        self._camera_ax = camera_ax = self._ros_fig.add_subplot(grid[:2, :2])
        self._depth_camera_ax = depth_camera_ax = self._ros_fig.add_subplot(grid[2:, :2])
        self._laser_ax = laser_ax = self._ros_fig.add_subplot(grid[:, 2:])

        camera_ax.axis('off')
        if self.gray_camera:
            self._ros_im = camera_ax.imshow(np.zeros((self.camera_height, self.camera_width),
                                                     dtype=np.uint8),
                                            vmin=0, vmax=255, cmap='gray',
                                            animated=True)
        else:
            self._ros_im = camera_ax.imshow(np.zeros((self.camera_height, self.camera_width, 3),
                                                     dtype=np.uint8),
                                            animated=True)
        depth_camera_ax.axis('off')
        self._ros_depth_im = depth_camera_ax.imshow(np.zeros((self.depth_camera_height, self.depth_camera_width),
                                                             dtype=np.uint8),
                                                    vmin=0, vmax=255, cmap='gray',
                                                    animated=True)

        laser_ax.spines['right'].set_visible(False)
        laser_ax.spines['top'].set_visible(False)
        laser_ax.set_xlim(-3, 3)
        laser_ax.set_ylim(-3, 3)
        laser_ax.invert_xaxis()
        self._ros_pos_hl, = laser_ax.plot([], [], linewidth=3, color='r')
        self._ros_pos_hl_sc = laser_ax.scatter([], [], s=50, color='r', marker='o')
        self._ros_vel_hl, = laser_ax.plot([], [], color='y')
        self._ros_laser_sc = laser_ax.scatter([], [], s=5, color='b')

        plt.show(block=False)

        plt.pause(0.1)

        self._bg = self._ros_fig.canvas.copy_from_bbox(self._ros_fig.bbox)

    def update_ros_plt(self):
        """
        更新ROS原始传感数据的可视化界面
        """

        self._ros_fig.canvas.restore_region(self._bg)

        # CAMERA
        self._ros_im.set_data(self.camera_image)
        self._camera_ax.draw_artist(self._ros_im)

        # DEPTH_CAMERA
        self._ros_depth_im.set_data(self.depth_camera_image)
        self._depth_camera_ax.draw_artist(self._ros_depth_im)

        # POSE
        if self.abs_position is None:  # USE RELATIVE POSE
            yaw = self.yaw - self.init_yaw

            position = self.position - self.init_position
            rotate_matrix = np.array([[math.cos(self.init_yaw), -math.sin(self.init_yaw)],
                                      [math.sin(self.init_yaw), math.cos(self.init_yaw)]])
            position = np.dot(position, rotate_matrix)
        else:  # USE ABSOLUTE POSE
            yaw = self.yaw
            position = self.abs_position[:2]

        d = 0.5
        pos_yaw = np.array([math.cos(yaw) * d, math.sin(yaw) * d]) + position
        pos_x, pos_y = position
        pos_yaw_x, pos_yaw_y = pos_yaw
        pos_x_converted, pos_y_converted = ros_xy_to_plt(pos_x, pos_y)
        pos_yaw_x_converted, pos_yaw_y_converted = ros_xy_to_plt(pos_yaw_x, pos_yaw_y)
        self._ros_pos_hl.set_xdata([pos_x_converted, pos_yaw_x_converted])
        self._ros_pos_hl.set_ydata([pos_y_converted, pos_yaw_y_converted])
        self._ros_pos_hl_sc.set_offsets([pos_x_converted, pos_y_converted])

        vel_x, vel_y = self.velocity
        vel_x_converted, vel_y_converted = ros_xy_to_plt(vel_x, vel_y)
        self._ros_vel_hl.set_xdata([0, vel_x_converted])
        self._ros_vel_hl.set_ydata([0, vel_y_converted])
        self._laser_ax.draw_artist(self._ros_pos_hl)
        self._laser_ax.draw_artist(self._ros_pos_hl_sc)
        self._laser_ax.draw_artist(self._ros_vel_hl)

        # LASER
        laser_scan, laser_scan_intensity = self._get_processed_laser_scan()
        laser_scan = [d if i > 1000 else 1000 for d, i in zip(laser_scan, laser_scan_intensity)]
        laser_scan = np.array(laser_scan)
        laser_rad = np.linspace(self.laser_scan_min_rad + yaw,
                                self.laser_scan_max_rad + yaw,
                                LASER_SCAN_SIZE)
        laser_x = np.cos(laser_rad) * laser_scan + pos_x
        laser_y = np.sin(laser_rad) * laser_scan + pos_y
        laser_x_converted, laser_y_converted = ros_xy_to_plt(laser_x, laser_y)
        self._ros_laser_sc.set_offsets(np.c_[laser_x_converted, laser_y_converted])
        self._laser_ax.draw_artist(self._ros_laser_sc)

        self._ros_fig.canvas.blit(self._ros_fig.bbox)

        self._ros_fig.canvas.flush_events()

    def stop(self):
        """
        强制停止小车
        """
        self.move_ugv(0, 0)

    def reset(self):
        """
        重置小车的初始状态
        """
        self.init_position = self.position
        self.init_yaw = self.yaw
        self.stop()

    def is_shutdown(self):
        return rospy.is_shutdown()

    def close(self, reason=''):
        """
        关闭ROS数据接收
        """
        self._closed = True
        rospy.signal_shutdown(reason)


if __name__ == "__main__":
    ugv = RosUGV(active_rl_keys_list=[RL_CAMERA, RL_DEPTH, RL_LASER, RL_STATE])

    while not ugv.initialized():
        ugv.display_status()
        time.sleep(0.5)
    ugv.display_status()

    print('ugv initialized')
    print('camera_HW', ugv.camera_height, ugv.camera_width)
    print('depth_camera_HW', ugv.depth_camera_height, ugv.depth_camera_width)
    print('laser_scan_size', len(ugv.laser_scan))
    if LASER_SCAN_SIZE != len(ugv.laser_scan):
        print(f'warning! laser_scan_size != {LASER_SCAN_SIZE}')
    print('laser_scan_min_rad', ugv.laser_scan_min_rad)
    print('laser_scan_max_rad', ugv.laser_scan_max_rad)

    ugv.init_ros_plt()

    while not ugv.is_shutdown():
        ugv.update_ros_plt()
