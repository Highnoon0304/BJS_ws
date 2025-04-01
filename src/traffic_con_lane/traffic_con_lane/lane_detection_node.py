#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import yaml
import open3d as o3d
import random

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # ------------------------------------------------
        # (1) 파라미터 설정
        # ------------------------------------------------
        self.declare_parameter('camera_calib_file', '/home/lee/Desktop/BJS_ws/src/traffic_con_lane/config/camera_intrinsic_calibration.yaml')
        self.declare_parameter('lidar_calib_file',  '/home/lee/Desktop/BJS_ws/src/traffic_con_lane/config/camera_extrinsic_calibration.yaml')
        
        camera_calib_file = self.get_parameter('camera_calib_file').get_parameter_value().string_value
        lidar_calib_file  = self.get_parameter('lidar_calib_file').get_parameter_value().string_value

        # 카메라 & 라이다 토픽
        self.image_topic = '/usb_cam_1/image_raw'
        self.lidar_topic = '/velodyne_points'

        # 구독/퍼블리셔
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.lidar_sub = self.create_subscription(PointCloud2, self.lidar_topic, self.lidar_callback, 10)
        self.path_pub  = self.create_publisher(Path, '/central_path', 10)

        # 내부 상태
        self.bridge       = CvBridge()
        self.latest_frame = None
        self.latest_cloud = None

        # 주기적 처리
        self.timer_ = self.create_timer(0.1, self.on_timer)  # 10Hz

        # ------------------------------------------------
        # (2) 캘리브레이션 로드
        # ------------------------------------------------
        self.camera_matrix, self.dist_coeffs = self.load_camera_calibration(camera_calib_file)
        self.T_lidar_to_cam = self.load_lidar_calibration(lidar_calib_file)

        # ------------------------------------------------
        # (3) HSV 범위 (사용자 지정)
        # ------------------------------------------------
        # 파란색: H=90~130, S=70~255, V=70~255
        self.lower_blue   = np.array([90, 70, 70], dtype=np.uint8)
        self.upper_blue   = np.array([130, 255, 255], dtype=np.uint8)
        # 노란색: H=15~45, S=70~255, V=70~255
        self.lower_yellow = np.array([15, 70, 70], dtype=np.uint8)
        self.upper_yellow = np.array([45, 255, 255], dtype=np.uint8)

        # 모폴로지 커널 (작업에 따라 튜닝)
        self.kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

        # ------------------------------------------------
        # (4) 라이다 ROI & DBSCAN 파라미터
        # ------------------------------------------------
        self.voxel_size         = 0.02
        self.lidar_roi_max_dist = 20.0
        self.x_min, self.x_max  = 0.0, 20.0
        self.y_min, self.y_max  = -5.0, 5.0
        self.z_min, self.z_max  = -1.0, 3.0

        self.dbscan_eps         = 0.5
        self.dbscan_min_points  = 5
        self.min_cluster_size   = 5

        # 추가: 전체 cone 포인트가 일정 개수 이상일 때만 클러스터링 진행
        self.min_total_points = 20

        # ------------------------------------------------
        # (5) 경로 생성 관련 파라미터
        # ------------------------------------------------
        # 선형회귀 기반 경로 (좌/우)
        self.prev_left_line  = None
        self.prev_right_line = None
        self.alpha_line      = 0.5
        self.line_max_x      = 6.0
        self.line_steps      = 20

        # 직접 접근 경로: (0,0,0)에서 좌/우 평균 중앙까지
        self.num_direct_steps = 20
        self.prev_target      = None
        self.alpha_target     = 0.4

        # Blending 비율
        self.alpha_blend = 0.5

        # ------------------------------------------------
        # (6) OpenCV 윈도우 설정
        # ------------------------------------------------
        cv2.namedWindow("ROI Masked", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Fused Sensor Data", cv2.WINDOW_NORMAL)

        self.get_logger().info("LaneDetectionNode started with specified HSV and clustering threshold.")

    # =====================================================
    # 캘리브레이션 함수
    # =====================================================
    def load_camera_calibration(self, calib_file):
        try:
            with open(calib_file, 'r') as f:
                data = yaml.safe_load(f)
            cm = np.array(data['camera_matrix']['data']).reshape(3,3)
            dist = np.array(data['distortion_coefficients']['data'])
            self.get_logger().info("Loaded camera calibration.")
            return cm, dist
        except Exception as e:
            self.get_logger().error(f"Failed to load camera calibration: {e}")
            return None, None

    def load_lidar_calibration(self, calib_file):
        try:
            with open(calib_file, 'r') as f:
                data = yaml.safe_load(f)
            if 'T_lidar_to_cam' in data:
                T = np.array(data['T_lidar_to_cam']).reshape(4,4)
            elif 'extrinsic_matrix' in data:
                T = np.array(data['extrinsic_matrix']).reshape(4,4)
            else:
                raise KeyError("No valid extrinsic key found.")
            self.get_logger().info("Loaded LiDAR calibration.")
            return T
        except Exception as e:
            self.get_logger().error(f"Failed to load LiDAR calibration: {e}")
            return None

    # =====================================================
    # 이미지 및 라이다 콜백
    # =====================================================
    def image_callback(self, msg: Image):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def lidar_callback(self, msg: PointCloud2):
        pts = []
        for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])
        if pts:
            self.latest_cloud = np.array(pts, dtype=np.float32)

    # =====================================================
    # 주기적 처리
    # =====================================================
    def on_timer(self):
        if self.latest_frame is None or self.latest_cloud is None:
            return

        # (A) 카메라 전체 영상 -> HSV 필터 적용
        frame = self.latest_frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_blue   = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_cone   = cv2.bitwise_or(mask_blue, mask_yellow)
        mask_cone   = cv2.morphologyEx(mask_cone, cv2.MORPH_OPEN,  self.kernel_open)
        mask_cone   = cv2.morphologyEx(mask_cone, cv2.MORPH_CLOSE, self.kernel_close)
        
        # (B) ROI Masked 창: HSV 마스크 적용 결과를 컬러 영상으로 표시
        roi_masked = cv2.bitwise_and(frame, frame, mask=mask_cone)

        # (C) 라이다 처리 및 클러스터링
        fused_frame = frame.copy()
        left_centers, right_centers = self.lidar_process(mask_cone, fused_frame)

        # (D) 경로 생성
        # 선형회귀 기반 경로
        left_line  = self.fit_line_ema(left_centers, is_left=True)
        right_line = self.fit_line_ema(right_centers, is_left=False)
        line_path_3d = self.create_line_path(left_line, right_line)

        # 직접 접근 경로: (0,0,0)->좌/우 평균 중앙
        new_target   = self.compute_target(left_centers, right_centers)
        final_target = self.smooth_target(new_target)
        direct_path_3d = self.create_direct_path(final_target)

        # Blending 두 경로
        final_path_3d = self.blend_paths(line_path_3d, direct_path_3d)

        # (E) 경로 시각화 (빨간 점)
        if final_path_3d:
            for p3d in final_path_3d:
                uv = self.project_lidar_to_image(p3d)
                if uv is not None:
                    cv2.circle(fused_frame, uv, 4, (0,0,255), -1)
            self.publish_path(final_path_3d, frame_id="velodyne")
        else:
            self.publish_empty_path()

        # (F) 디버그 창: "ROI Masked"에는 HSV 마스크 결과, "Fused Sensor Data"에는 최종 합성 결과
        cv2.imshow("ROI Masked", roi_masked)
        cv2.imshow("Fused Sensor Data", fused_frame)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()

    # =====================================================
    # 라이다 처리: ROI, DBSCAN, 좌/우 분류
    # =====================================================
    def lidar_process(self, mask_cone, fused_frame):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.latest_cloud)
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        pts_down = np.asarray(pcd_down.points, dtype=np.float32)

        pts_roi = self.apply_lidar_roi_filter(pts_down)
        if pts_roi.size == 0:
            return [], []

        h, w = fused_frame.shape[:2]
        cone_points = []
        for p in pts_roi:
            uv = self.project_lidar_to_image(p)
            if uv is None:
                continue
            u, v = uv
            if 0 <= u < w and 0 <= v < h:
                if mask_cone[v, u] != 0:
                    cone_points.append(p)

        # 만약 전체 cone 포인트 수가 너무 적으면 클러스터링하지 않음 (튜닝용)
        if len(cone_points) < self.min_total_points:
            return [], []

        for cp in cone_points:
            uv2 = self.project_lidar_to_image(cp)
            if uv2 is not None:
                cv2.circle(fused_frame, uv2, 3, (0,255,255), -1)

        pcd_cones = o3d.geometry.PointCloud()
        pcd_cones.points = o3d.utility.Vector3dVector(np.array(cone_points, dtype=np.float32))
        labels = np.array(pcd_cones.cluster_dbscan(eps=self.dbscan_eps, min_points=self.dbscan_min_points, print_progress=False))
        max_label = labels.max()
        if max_label < 0:
            return [], []

        cone_np = np.array(cone_points, dtype=np.float32)
        left_centers = []
        right_centers = []
        for cid in range(max_label+1):
            idx = np.where(labels == cid)[0]
            cluster_pts = cone_np[idx]
            if len(cluster_pts) < self.min_cluster_size:
                continue
            center_3d = np.mean(cluster_pts, axis=0)
            col = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            for pt in cluster_pts:
                uv3 = self.project_lidar_to_image(pt)
                if uv3 is not None:
                    cv2.circle(fused_frame, uv3, 2, col, -1)
            dist = np.linalg.norm(center_3d)
            uv_c = self.project_lidar_to_image(center_3d)
            if uv_c is not None:
                cv2.putText(fused_frame, f"{dist:.2f}m", (uv_c[0]+5, uv_c[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            if center_3d[1] < 0:
                left_centers.append(center_3d)
            else:
                right_centers.append(center_3d)

        return left_centers, right_centers

    # =====================================================
    # 선형회귀 + EMA (좌/우)
    # =====================================================
    def fit_line_ema(self, centers, is_left=True):
        if len(centers) < 1:
            return None
        elif len(centers) == 1:
            x0, y0, _ = centers[0]
            line_raw = (0.0, y0)
        else:
            xs = np.array([c[0] for c in centers])
            ys = np.array([c[1] for c in centers])
            a, b = np.polyfit(xs, ys, 1)
            line_raw = (a, b)
        old_line = self.prev_left_line if is_left else self.prev_right_line
        if old_line is None:
            if is_left:
                self.prev_left_line = line_raw
            else:
                self.prev_right_line = line_raw
            return line_raw
        a_old, b_old = old_line
        a_new, b_new = line_raw
        a_final = self.alpha_line * a_new + (1 - self.alpha_line) * a_old
        b_final = self.alpha_line * b_new + (1 - self.alpha_line) * b_old
        if is_left:
            self.prev_left_line = (a_final, b_final)
        else:
            self.prev_right_line = (a_final, b_final)
        return (a_final, b_final)

    # =====================================================
    # 선형회귀 기반 경로 생성
    # =====================================================
    def create_line_path(self, left_line, right_line):
        path_points = []
        if left_line is None and right_line is None:
            return path_points
        if left_line is None:
            a_r, b_r = right_line
            for i in range(self.line_steps + 1):
                t = i / float(self.line_steps)
                x_val = t * self.line_max_x
                y_r = a_r * x_val + b_r
                path_points.append((x_val, y_r, 0.0))
            return path_points
        if right_line is None:
            a_l, b_l = left_line
            for i in range(self.line_steps + 1):
                t = i / float(self.line_steps)
                x_val = t * self.line_max_x
                y_l = a_l * x_val + b_l
                path_points.append((x_val, y_l, 0.0))
            return path_points
        aL, bL = left_line
        aR, bR = right_line
        for i in range(self.line_steps + 1):
            t = i / float(self.line_steps)
            x_val = t * self.line_max_x
            yL = aL * x_val + bL
            yR = aR * x_val + bR
            mid_y = (yL + yR) / 2.0
            path_points.append((x_val, mid_y, 0.0))
        return path_points

    # =====================================================
    # 직접 접근 경로 생성: (0,0,0) -> 타겟
    # =====================================================
    def create_direct_path(self, final_target):
        path_points = []
        if final_target is None:
            return path_points
        for i in range(self.num_direct_steps + 1):
            t = i / float(self.num_direct_steps)
            px = t * final_target[0]
            py = t * final_target[1]
            pz = t * final_target[2]
            path_points.append((px, py, pz))
        return path_points

    # =====================================================
    # 두 경로 Blending
    # =====================================================
    def blend_paths(self, line_path, direct_path):
        if len(line_path) < 2 and len(direct_path) < 2:
            return []
        if len(line_path) < 2:
            return direct_path
        if len(direct_path) < 2:
            return line_path
        final_path = []
        n_min = min(len(line_path), len(direct_path))
        alpha = self.alpha_blend
        for i in range(n_min):
            lx, ly, lz = line_path[i]
            dx, dy, dz = direct_path[i]
            fx = alpha * lx + (1 - alpha) * dx
            fy = alpha * ly + (1 - alpha) * dy
            fz = alpha * lz + (1 - alpha) * dz
            final_path.append((fx, fy, fz))
        return final_path

    # =====================================================
    # 좌/우 평균 -> 중앙 타겟 계산
    # =====================================================
    def compute_target(self, left_centers, right_centers):
        if len(left_centers) == 0 and len(right_centers) == 0:
            return None
        c_left = self.mean_point(left_centers)
        c_right = self.mean_point(right_centers)
        if c_left is not None and c_right is not None:
            cx_mid = (c_left[0] + c_right[0]) / 2.0
            cy_mid = (c_left[1] + c_right[1]) / 2.0
            cz_mid = (c_left[2] + c_right[2]) / 2.0
            return (cx_mid, cy_mid, cz_mid)
        elif c_left is not None:
            return c_left
        elif c_right is not None:
            return c_right
        else:
            return None

    def smooth_target(self, new_target):
        if new_target is None:
            return None
        if self.prev_target is None:
            self.prev_target = new_target
            return new_target
        alpha = self.alpha_target
        ox, oy, oz = self.prev_target
        nx, ny, nz = new_target
        fx = alpha * nx + (1 - alpha) * ox
        fy = alpha * ny + (1 - alpha) * oy
        fz = alpha * nz + (1 - alpha) * oz
        final_target = (fx, fy, fz)
        self.prev_target = final_target
        return final_target

    # =====================================================
    # 보조 함수들
    # =====================================================
    def mean_point(self, pts):
        if len(pts) == 0:
            return None
        arr = np.array(pts, dtype=np.float32)
        return tuple(np.mean(arr, axis=0))

    def apply_lidar_roi_filter(self, pts):
        dist2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
        mask_dist = dist2 < (self.lidar_roi_max_dist ** 2)
        mask_x = (pts[:, 0] >= self.x_min) & (pts[:, 0] <= self.x_max)
        mask_y = (pts[:, 1] >= self.y_min) & (pts[:, 1] <= self.y_max)
        mask_z = (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max)
        mask = mask_dist & mask_x & mask_y & mask_z
        return pts[mask]

    def project_lidar_to_image(self, pt3d):
        if self.T_lidar_to_cam is None:
            return None
        pt_h = np.array([pt3d[0], pt3d[1], pt3d[2], 1.0], dtype=np.float32)
        cam_pt = self.T_lidar_to_cam @ pt_h
        if cam_pt[2] <= 0:
            return None
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        u = int(fx * cam_pt[0] / cam_pt[2] + cx)
        v = int(fy * cam_pt[1] / cam_pt[2] + cy)
        return (u, v)

    def publish_path(self, path_points_3d, frame_id="velodyne"):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = frame_id
        for (px, py, pz) in path_points_3d:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(px)
            pose.pose.position.y = float(py)
            pose.pose.position.z = float(pz)
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def publish_empty_path(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "velodyne"
        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("lane_detection_node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

