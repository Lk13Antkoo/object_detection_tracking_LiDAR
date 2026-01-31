import pandas as pd
import open3d
import numpy as np
from utilities.classification import classify_from_obb_and_plane
from utilities.obb import get_horizontal_obb
from utilities.draw_id import *
from os import path, name
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)
from math import dist
from tabnanny import check
from matplotlib.pyplot import flag

# 4. Load point cloud and segment into clusters
def load_clustred_point_cloud(path_to_pcd):
    # 1. Load Data
    pcd = open3d.io.read_point_cloud(path_to_pcd)
    # open3d.visualization.draw_geometries([pcd])

    # 2. Voxel downsampling
    #print(f"Points before downsampling: {len(pcd.points)} ")
    pcd = pcd.voxel_down_sample(voxel_size=0.1)

    # 3. RANSAC Segmentation to identify the largest plane (in this case the ground/road) from objects above it
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=150)
    ## Identify inlier points -> road
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0, 1, 1])

    ## Identify outlier points -> objects on the road
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0])

    # 4. Clustering using DBSCAN -> To further segment objects on the road
    with open3d.utility.VerbosityContextManager(open3d.utility.VerbosityLevel.Error) as cm:
        
        labels = np.array(outlier_cloud.cluster_dbscan(eps=0.4, min_points=7, print_progress=False))

    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels/(max_label if max_label>0 else 1))
    colors[labels<0] = 0
    ## Colorized objects on the road
    outlier_cloud.colors = open3d.utility.Vector3dVector(colors[:, :3])

    return plane_model, inlier_cloud, outlier_cloud, labels

# 5. Generate 3D Bounding Boxes and filter boxes based on criteria

def filter_box(plane_model, inlier_cloud, outlier_cloud, labels):

    dataframe_info = pd.DataFrame(columns=['object_id', 'x', 'y', 'z'])

    filtered_obbs = []
    indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()
    MAX_POINTS = 1500
    MIN_POINTS = 20
    a, b, c, d = plane_model
    
    normal = np.array([a, b, c])
    MAX_HEIGHT = 2   # meters
    X_positive = 1.2
    X_negative = -8
    Y_positive = 14.5
    object_number = 0
    ## For each individual object on the road
    for i in range(0, len(indexes)):
        nb_points = len(outlier_cloud.select_by_index(indexes[i]).points)
        # If object size within the criteria, draw bounding box
        if (nb_points>MIN_POINTS and nb_points<MAX_POINTS):
            sub_cloud = outlier_cloud.select_by_index(indexes[i])
            obb = get_horizontal_obb(sub_cloud)
                #=== Filter boxes based on position
            center = obb.get_center()

            if ((center[0] < X_positive) and (center[0] > X_negative)) or (center[1] < Y_positive):
                flag_position = True
            else:
                flag_position = False

            min_bound = obb.get_min_bound()   # lowest corner [x,y,z]
            max_bound = obb.get_max_bound() 

            checking_height = abs(max_bound[2] - min_bound[2])

            dist_max_bound = abs(np.dot(normal, max_bound) + d) / np.linalg.norm(normal)
            dist_min_bound = abs(np.dot(normal, min_bound) + d) / np.linalg.norm(normal)
        # Keep only boxes close to the plane and not too tall
            #if  height_box < HEIGHT_BOX_MAX and checking_height < 2.5 and (min_bound[2] < 1) and  height_box > HEIGHT_BOX_MIN: #and height_box > HEIGHT_BOX_MIN: #and dist_max_bound > MIN_HEIGHT, dist_max_bound < MAX_HEIGHT  and
            if  (dist_max_bound > 0.4) and (dist_max_bound <MAX_HEIGHT) and (checking_height < 2.5) and (checking_height > 0.1) and flag_position: #and (dist_min_bound < MIN_HEIGHT):  #and (checking_height > 0.1): #and height_box > HEIGHT_BOX_MIN: #and dist_max_bound > MIN_HEIGHT, dist_max_bound < MAX_HEIGHT  and
            
                filtered_obbs.append(obb)
                #=== calculate the mean value of the object points
                pts = np.asarray(sub_cloud.points)
                mean_xyz = pts.mean(axis=0)
                object_number += 1
                dataframe_info.loc[len(dataframe_info)] = {'object_id': object_number,
                                                        'x': mean_xyz[0],
                                                        'y': mean_xyz[1],
                                                        'z': mean_xyz[2],}

    ## Combined all visuals: outlier_cloud (objects), obbs (oriented bounding boxes), inlier_cloud (road)
    list_of_visuals = []
    list_of_visuals.append(outlier_cloud)
    #list_of_visuals.extend(obbs)
    list_of_visuals.extend(filtered_obbs)
    #list_of_visuals.extend(labels_3d)

    #== road
    list_of_visuals.append(inlier_cloud)


    #print(type(list_of_visuals))

    return list_of_visuals, dataframe_info,filtered_obbs

# Load CSV
def save_temp_pcd(id_frame):

    df = pd.read_csv(f"./data_IU_final/raw_data_consecutive/192.168.26.26_2020-11-25_20-01-45_frame-{id_frame}.csv", sep=';')

    # Extract XYZ
    points = df[['X', 'Y', 'Z']].to_numpy()
    # Create Open3D point cloud
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    # Save as PCD
    open3d.io.write_point_cloud(f"./data_IU_final/temporal_pcd_process/{id_frame}.pcd", pcd)
    return f"./data_IU_final/temporal_pcd_process/{id_frame}.pcd"

# Track ID of objects across frames
def assignIds(prevDf, currDf, next_id, maxDistance=4.0):
    nextId = next_id
    prevCentroids = prevDf[['x', 'y', 'z']].to_numpy()
    currCentroids = currDf[['x', 'y', 'z']].to_numpy()

    prevLabels = prevDf["object_id"].tolist()
    currLabels = currDf["object_id"].tolist()

    costMatrix = cdist(prevCentroids, currCentroids)

    rowInd, colInd = linear_sum_assignment(costMatrix)

    #nextId = max(prevLabels) + 1 if  len(prevLabels)!=0 else 0

    #print("check  max _ pre ID:, ", nextId)

    temp_out_label = [-1 for _ in range(len(currLabels))]
    
    for r, c in zip(rowInd, colInd):
        if costMatrix[r, c] < maxDistance:  
            temp_out_label[c] = prevLabels[r]
        

    for i in range(len(temp_out_label)):
        if temp_out_label[i] == -1:
            temp_out_label[i] = nextId
            nextId += 1
    currLabels = temp_out_label
    
    return currLabels, nextId

def main():
    # 1. Define frames to process
    list_frame = list(range(1849, 2567))

    previous_dataframe_info = pd.DataFrame(columns=['object_id', 'x', 'y', 'z'])
    global_id_object = 0

    print("Starting processing frames...")
    for i in tqdm(list_frame):
        current_path_pcd = save_temp_pcd(i)
        current_plane_model, current_inlier_cloud, current_outlier_cloud, current_labels = load_clustred_point_cloud(current_path_pcd)
        # list_of_visuals contains class, bounding box, and point cloud
        # dataframe_info contains object_id and its mean x,y,z
        current_list_of_visuals, current_dataframe_info, current_filtered_obbs = filter_box(current_plane_model, current_inlier_cloud, current_outlier_cloud, current_labels)

        if i == list_frame[0]:
            current_list_id = current_dataframe_info["object_id"].tolist()
            global_id_object = max(current_list_id) + 1 if len(current_list_id)>0 else 0
        else:
            current_list_id, next_id_check = assignIds(previous_dataframe_info, current_dataframe_info, global_id_object)
            if next_id_check > global_id_object:
                global_id_object = next_id_check 
            
        previous_dataframe_info = current_dataframe_info.copy()
        previous_dataframe_info["object_id"] = current_list_id
    #===========================================
        labels_str = []
        for index, obb in enumerate(current_filtered_obbs):
            
            name = classify_from_obb_and_plane(obb, current_plane_model) + str(current_list_id[index])
            labels_str.append(name)
        visualize_matplotlib(current_list_of_visuals, labels_str, "./data_IU_final/final_result/", frameId=i)
    #===========================================
    print("Processing completed.")

if __name__ == "__main__":
    main()
