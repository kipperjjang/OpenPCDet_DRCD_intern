import re
import os
import time
import pickle
import numpy as np
import open3d as o3d


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

def visualize_lidar_data(num, data):
    boxes = data[num]['boxes_lidar']
    labels = data[num]['pred_labels']
    frame_id = data[num]['frame_id']
    frame_group_name = extract_between_underscores(frame_id)
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

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    ch_data_path = os.path.expanduser(f"~/OpenPCDet/data/nuscenes/v1.0-mini/samples/{frame_group_name}")
    if os.path.exists(ch_data_path):
        if frame_group_name == 'LIDAR_TOP':
            bin_data_path = os.path.expanduser(f"~/OpenPCDet/data/nuscenes/v1.0-mini/samples/{frame_group_name}/{frame_id}.bin")
            pcd = load_bin_point_cloud(bin_data_path)
            print(pcd)
        else:
            pcd_data_path = os.path.expanduser(f"~/OpenPCDet/data/nuscenes/v1.0-mini/samples/{frame_group_name}/{frame_id}")
            pcd = o3d.io.read_point_cloud(pcd_data_path)
        vis.add_geometry(pcd)
    else:
        print(f"PCD file not found: {ch_data_path}")
        return

    #include bbox in vis
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
        bbox.rotate(rotation_matrix, center = bbox.center)
        vis.add_geometry(bbox)

    opt = vis.get_render_option()
    opt.point_size = 3.0

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # file_path = "sample/transfusion_lidar_result.pkl"
    file_path = "sample/voxelnext_result.pkl"
    # file_path = "sample/vn_mini_result.pkl"
    num = int(input("# of data : "))
    result_data = load_pkl(file_path)
    visualize_lidar_data(num, result_data)
    # for i in range(80):
    #     visualize_lidar_data(i, result_data)
