import re
import os
import time
import pickle
import numpy as np
import open3d as o3d
import cv2
from screeninfo import get_monitors

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def extract_between_underscores(text):
    match = re.search(r'__(.*?)__', text)
    return match.group(1) if match else None

def load_bin_point_cloud(bin_path):
    if not os.path.exists(bin_path):
        print(f"Error: File not found - {bin_path}")
        return None

    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd

def find_img_file_name(key1, key2, image_path):
    for filename in os.listdir(image_path):
        if os.path.isfile(os.path.join(image_path, filename)):
            if key1 == filename[0:29] and key2 == filename[-11:-9]:
                return filename

def get_screen_resolution():
    monitor = get_monitors()[0]
    return monitor.width, monitor.height

def visualization(data):
    colors = {
        1: [1, 0, 0],  # car - red
        2: [0, 1, 0],  # truck - green
        4: [0, 0, 1],  # bus - blue
        5: [1, 1, 0],  # trailer - yellow
        6: [1, 0, 1],  # barrier - magenta
        7: [0, 1, 1],  # bicycle - cyan
        8: [0.5, 0.5, 0],  # motorcycle - olive
        9: [0.5, 0, 0.5],  # pedestrian - purple
        10: [0, 0.5, 0.5]  # traffic_cone - teal
    }

    width, height = 1920, 1080

    vis = o3d.visualization.Visualizer()
    vis.create_window("PCD", width // 2, height)

    cv2.namedWindow("Image Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image Viewer", width // 2, height)

    opt = vis.get_render_option()
    opt.point_size = 3.0

    pcd = o3d.geometry.PointCloud()
    bbox_list = []

    for frame in range(81):
        print(f"Frame: {frame}")

        boxes = data[frame]['boxes_lidar']
        labels = data[frame]['pred_labels']
        frame_id = data[frame]['frame_id']
        image_path_1 = os.path.expanduser(f"~/OpenPCDet/data/nuscenes/v1.0-mini/samples/CAM_FRONT/")
        image_path_2 = os.path.expanduser(f"~/OpenPCDet/data/nuscenes/v1.0-mini/samples/CAM_BACK/")
        key1 = frame_id[0:29]
        key2 = frame_id[-11:-9]

        frame_group_name = extract_between_underscores(frame_id)

        ch_data_path = os.path.expanduser(f"~/OpenPCDet/data/nuscenes/v1.0-mini/samples/{frame_group_name}")
        if os.path.exists(ch_data_path):
            if frame_group_name == 'LIDAR_TOP':
                bin_data_path = os.path.expanduser(
                    f"~/OpenPCDet/data/nuscenes/v1.0-mini/samples/{frame_group_name}/{frame_id}.bin")
                new_pcd = load_bin_point_cloud(bin_data_path)
            else:
                pcd_data_path = os.path.expanduser(
                    f"~/OpenPCDet/data/nuscenes/v1.0-mini/samples/{frame_group_name}/{frame_id}")
                new_pcd = o3d.io.read_point_cloud(pcd_data_path)
            pcd.points = new_pcd.points
            if frame == 0:
                vis.add_geometry(pcd)
            else:
                vis.update_geometry(pcd)
        else:
            print(f"PCD file not found: {ch_data_path}")
            return

        for bbox in bbox_list:
            vis.remove_geometry(bbox)
        bbox_list.clear()

        # include bbox in vis
        for i, box in enumerate(boxes):
            center = box[:3]
            size = box[3:6]
            rotation = np.radians(box[6])
            label = labels[i]
            color = colors.get(label, [0.5, 0.5, 0.5])

            bbox = o3d.geometry.OrientedBoundingBox()
            bbox.center = center
            bbox.extent = size
            bbox.color = color
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, rotation])
            bbox.rotate(rotation_matrix, center=bbox.center)

            bbox_list.append(bbox)
            vis.add_geometry(bbox)

        ctr = vis.get_view_control()
        ctr.set_lookat([0, 0, 0])
        ctr.set_zoom(0.5)

        image_path_front = image_path_1 + find_img_file_name(key1, key2, image_path_1)
        image_path_back = image_path_2 + find_img_file_name(key1, key2, image_path_2)

        image1 = cv2.imread(image_path_front)
        image2 = cv2.imread(image_path_back)

        combined_image = cv2.vconcat([image1, image2])

        cv2.imshow("Image Viewer", combined_image)

        vis.poll_events()
        vis.update_renderer()

        cv2.waitKey(500)

    vis.destroy_window()
    cv2.destroyWindow("Image Viewer")

def visualization_manual(data):
    def visualize_images(image_path_front, image_path_back):
        image1 = cv2.imread(image_path_front)
        image2 = cv2.imread(image_path_back)

        combined_image = cv2.vconcat([image1, image2])

        combined_image = np.asarray(combined_image)

        vis_image = o3d.visualization.Visualizer()
        vis_image.create_window("Image Viewer", width // 2, height)

        o3d_image = o3d.geometry.Image(combined_image)
        vis_image.add_geometry(o3d_image)

        vis_image.run()

    colors = {
        1: [1, 0, 0],  # car - red
        2: [0, 1, 0],  # truck - green
        4: [0, 0, 1],  # bus - blue
        5: [1, 1, 0],  # trailer - yellow
        6: [1, 0, 1],  # barrier - magenta
        7: [0, 1, 1],  # bicycle - cyan
        8: [0.5, 0.5, 0],  # motorcycle - olive
        9: [0.5, 0, 0.5],  # pedestrian - purple
        10: [0, 0.5, 0.5]  # traffic_cone - teal
    }

    width, height = get_screen_resolution()

    vis = o3d.visualization.Visualizer()
    vis.create_window("PCD", width//2, height)

    opt = vis.get_render_option()
    opt.point_size = 3.0

    pcd = o3d.geometry.PointCloud()
    bbox_list = []

    for frame in range(81):
        print(f"Frame: {frame}")

        boxes = data[frame]['boxes_lidar']
        labels = data[frame]['pred_labels']
        frame_id = data[frame]['frame_id']
        image_path_1 = os.path.expanduser(f"~/OpenPCDet/data/nuscenes/v1.0-mini/samples/CAM_FRONT/")
        image_path_2 = os.path.expanduser(f"~/OpenPCDet/data/nuscenes/v1.0-mini/samples/CAM_BACK/")
        key1 = frame_id[0:29]
        key2 = frame_id[-11:-9]
        frame_group_name = extract_between_underscores(frame_id)

        ch_data_path = os.path.expanduser(f"~/OpenPCDet/data/nuscenes/v1.0-mini/samples/{frame_group_name}")
        if os.path.exists(ch_data_path):
            if frame_group_name == 'LIDAR_TOP':
                bin_data_path = os.path.expanduser(
                    f"~/OpenPCDet/data/nuscenes/v1.0-mini/samples/{frame_group_name}/{frame_id}.bin")
                new_pcd = load_bin_point_cloud(bin_data_path)
                # print(pcd)
            else:
                pcd_data_path = os.path.expanduser(
                    f"~/OpenPCDet/data/nuscenes/v1.0-mini/samples/{frame_group_name}/{frame_id}")
                new_pcd = o3d.io.read_point_cloud(pcd_data_path)
            pcd.points = new_pcd.points
            if frame == 0:
                vis.add_geometry(pcd)
            else:
                vis.update_geometry(pcd)
        else:
            print(f"PCD file not found: {ch_data_path}")
            return

        for bbox in bbox_list:
            vis.remove_geometry(bbox)
        bbox_list.clear()

        # include bbox in vis
        for i, box in enumerate(boxes):
            center = box[:3]
            size = box[3:6]
            rotation = np.radians(box[6])
            label = labels[i]
            color = colors.get(label, [0.5, 0.5, 0.5])

            bbox = o3d.geometry.OrientedBoundingBox()
            bbox.center = center
            bbox.extent = size
            bbox.color = color
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, rotation])
            bbox.rotate(rotation_matrix, center=bbox.center)

            bbox_list.append(bbox)
            vis.add_geometry(bbox)

        ctr = vis.get_view_control()
        ctr.set_lookat([0, 0, 0])
        ctr.set_zoom(0.5)

        vis.poll_events()
        vis.update_renderer()

        image_path_front = image_path_1 + find_img_file_name(key1, key2, image_path_1)
        image_path_back = image_path_2 + find_img_file_name(key1, key2, image_path_2)
        visualize_images(image_path_front, image_path_back)

        time.sleep(0.1)

    vis.destroy_window()

if __name__ == "__main__":
    # file_path = "sample/transfusion_lidar_result.pkl"
    file_path = "sample/voxelnext_result.pkl"
    # file_path = "sample/vn_mini_result.pkl"
    # num = int(input("# of data : "))
    result_data = load_pkl(file_path)

    visualization(result_data)

