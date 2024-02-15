from enum import IntEnum

import cv2
import pyrealsense2 as rs
import open3d as o3d
import numpy as np

def init_rs_camera(width, height):
    class Preset(IntEnum):
        Custom = 0
        Default = 1
        Hand = 2
        HighAccuracy = 3
        HighDensity = 4
        MediumDensity = 5

    pipeline = rs.pipeline()
    config = rs.config()

    FPS = 30
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, FPS)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    # Returns the depth scale in meters
    depth_scale_meters = depth_sensor.get_depth_scale()    # 0.001

    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Processing blocks
    pc = rs.pointcloud()
    colorizer = rs.colorizer()

    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intr = color_profile.get_intrinsics()

    intr_matr = np.zeros((3,3))
    intr_matr[0, 0] = color_intr.fx
    intr_matr[1, 1] = color_intr.fy
    intr_matr[0, 2] = color_intr.ppx
    intr_matr[1, 2] = color_intr.ppy
    dist_coeffs = np.asarray(color_intr.coeffs, dtype=np.float64)

    return pipeline, align, pc, colorizer, depth_scale_meters, intr_matr, dist_coeffs


def read_rs_camera(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

    if aligned_depth_frame and color_frame:
        color_frame = np.asanyarray(color_frame.get_data())
        depth_frame = np.asanyarray(aligned_depth_frame.get_data())
    else:
        color_frame, depth_frame = None, None

    return color_frame, depth_frame, intrinsics 


def make_point_cloud(color_bgr, depth_m, fx, fy, cx, cy, trunc_depth_values_m):
    color_o3d_img = o3d.geometry.Image(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
    depth_o3d_img = o3d.geometry.Image(depth_m)
    # see default values of the function below: the input depth_o3d_img image 
    # must be in millimeters, and we keep the same scale
    # (i.e. parameter depth_scale=1).
    rgbd_o3d_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d_img, depth_o3d_img, convert_rgb_to_intensity=False,
        depth_scale=1.0, depth_trunc=trunc_depth_values_m
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        color_bgr.shape[1], color_bgr.shape[0], fx, fy, cx, cy
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_o3d_img, intrinsic
    )

    return pcd


def init_3d_renderer(width, height, geometries):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    for g in geometries:
        vis.add_geometry(g)

    view = vis.get_view_control()
    view.set_lookat([0, 0, 0])
    view.set_up([0, -1, 0])
    view.set_front([0, 0, -1])
    view.set_zoom(0.1)
    view.set_constant_z_far(10000)
    
    return vis

    
def update_3d_renderer(vis, geometries):
    for g in geometries:
        vis.update_geometry(g)
    vis.poll_events()
    vis.update_renderer()