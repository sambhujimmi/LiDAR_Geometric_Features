import open3d as o3d
import numpy as np

def load_point_cloud(file_path):
    print(f"Reading point cloud from {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Point cloud loaded with {len(pcd.points)} points")
    return pcd

def preprocess_point_cloud(pcd, voxel_size=0.05, remove_outliers=True):
    # Downsample using voxel grid
    print(f"Downsampling with voxel size {voxel_size}")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    if remove_outliers:
        print("Removing outliers")
        # Statistical outlier removal
        pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
    print(f"Preprocessed point cloud has {len(pcd_down.points)} points")
    return pcd_down

def compute_normals(pcd, radius=0.1, max_nn=30):
    print("Computing surface normals")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=15)
    return pcd

def compute_local_pca(points, n_neighbors=30):
    pcd_tree = o3d.geometry.KDTreeFlann(points)
    point_array = np.asarray(points.points)
    
    eigenvalues = np.zeros((len(point_array), 3))
    eigenvectors = np.zeros((len(point_array), 3, 3))
    
    for i in range(len(point_array)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point_array[i], n_neighbors)
        neighbors = point_array[idx]
        
        # Center the neighborhood
        centered = neighbors - np.mean(neighbors, axis=0)
        
        # Compute covariance matrix
        cov = np.cov(centered.T)
        
        # Compute eigenvalues and eigenvectors
        w, v = np.linalg.eigh(cov)
        
        # Sort eigenvalues in descending order
        idx_sorted = np.argsort(w)[::-1]
        eigenvalues[i] = w[idx_sorted]
        eigenvectors[i] = v[:, idx_sorted]
    
    return eigenvalues, eigenvectors

def extract_eigenvalue_features(pcd, n_neighbors=30):
    print("Computing eigenvalues and extracting geometric features")
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = compute_local_pca(pcd, n_neighbors)
    
    # Initialize feature dictionary
    features = {}
    
    # Store raw eigenvalues
    features['eigenvalue_1'] = eigenvalues[:, 0]  # Largest eigenvalue
    features['eigenvalue_2'] = eigenvalues[:, 1]  # Middle eigenvalue
    features['eigenvalue_3'] = eigenvalues[:, 2]  # Smallest eigenvalue
    
    # Calculate derived features
    lambda1 = eigenvalues[:, 0]
    lambda2 = eigenvalues[:, 1]
    lambda3 = eigenvalues[:, 2]
    
    # Avoid division by zero
    # eps = 1e-10
    sum_lambda = lambda1 + lambda2 + lambda3 # + eps
    
    # 1. Omnivariance - geometric mean of eigenvalues
    features['omnivariance'] = np.cbrt(lambda1 * lambda2 * lambda3) # + eps
    
    # 2. Eigentropy - entropy of normalized eigenvalues
    features['eigentropy'] = -(lambda1 * np.log(lambda1)+ lambda2 * np.log(lambda2) + lambda3 * np.log(lambda3))
    
    # 3. Anisotropy - difference between largest and smallest eigenvalues
    features['anisotropy'] = (lambda1 - lambda3) / (lambda1) # + eps
    
    # 4. Planarity - describes how planar the local surface is
    features['planarity'] = (lambda2 - lambda3) / (lambda1) # + eps
    
    # 5. Linearity - describes how linear the local structure is
    features['linearity'] = (lambda1 - lambda2) / (lambda1) # + eps
    
    # 6. Surface variation - equivalent to "change of curvature"
    features['surface_variation'] = lambda3 / sum_lambda

    # 7. Sphericity - how spherical the local neighborhood is
    features['sphericity'] = lambda3 / (lambda1) # + eps
    
    # 8. Verticality - how vertical the local surface is
    # Using the smallest eigenvector as the normal direction
    normals = eigenvectors[:, :, 2] 
    features['verticality'] = 1-np.abs(normals[:, 2])
    
    return features

def visualize_point_cloud(pcd, feature=None, window_name="Point Cloud"):
    feature_pcd = o3d.geometry.PointCloud(pcd)
    if feature is not None:
        # Handle NaN or infinite values
        feature = np.nan_to_num(feature, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Normalize feature to [0, 1] for coloring
        feature_min = np.min(feature)
        feature_max = np.max(feature)
        
        if feature_max - feature_min > 1e-10:
            normalized_feature = (feature - feature_min) / (feature_max - feature_min)
        else:
            normalized_feature = np.zeros_like(feature)
        
        # Create color map
        colors = np.zeros((len(normalized_feature), 3))
        colors[:, 0] = 2*normalized_feature-1  # Red 
        colors[:, 1] = 1-2*np.abs(normalized_feature-0.5)  # Green 
        colors[:, 2] = 1 - 2*normalized_feature  # Blue 
        feature_pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(feature_pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    render_option.show_coordinate_frame = True
    render_option.light_on = True
    vis.set_view_status("""{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 91.323249816894531, 68.312934875488281, 239.36143493652344 ],
			"boundingbox_min" : [ -61.986682891845703, -86.929100036621094, 202.42361450195312 ],
			"field_of_view" : 60.0,
			"front" : [ 0.28811867783792167, -0.68276323493978641, 0.67143279075085849 ],
			"lookat" : [ 11.84579119083973, -26.865280518885033, 201.42609146080261 ],
			"up" : [ -0.35748233746646552, 0.57378659908016816, 0.73686858876975259 ],
			"zoom" : 0.53199999999999981
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}""")
    # vis.register_animation_callback(lambda vis: vis.get_view_control().rotate(-0.5, 0.3))  # Rotation animation
    vis.run()
    vis.destroy_window()

def main():
    file_path = "point_cloud_downsampled.pcd"

    # file_path = o3d.data.PLYPointCloud().path # Example PLY file from Open3D data
    
    # Load point cloud
    pcd = load_point_cloud(file_path)
    
    # Visualize original point cloud
    visualize_point_cloud(pcd, window_name="Original Point Cloud")

    # pcd.rotate(pcd.get_rotation_matrix_from_xyz((0, np.pi, np.pi)))
    # Visualize rotated point cloud
    # visualize_point_cloud(pcd, "Rotated Point Cloud")
    
    # Preprocess the point cloud
    # pcd_preprocessed = pcd
    pcd_preprocessed = preprocess_point_cloud(pcd, voxel_size=0.1)

    # Visualize preprocessed point cloud
    visualize_point_cloud(pcd_preprocessed, window_name="Preprocessed Point Cloud")
    
    # Compute normals
    pcd_with_normals = compute_normals(pcd_preprocessed)
    
    # Extract eigenvalue-based features
    features = extract_eigenvalue_features(pcd_with_normals)
    
    # Print statistics about the features
    for feature_name, feature_values in features.items():
        min_val = np.min(feature_values)
        max_val = np.max(feature_values)
        mean_val = np.mean(feature_values)
        median_val = np.median(feature_values)
        
        print(f"{feature_name}:")
        print(f"  Min: {min_val:.4f}")
        print(f"  Max: {max_val:.4f}")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Median: {median_val:.4f}")
        print()
        if 'eigenvalue' not in feature_name:
            visualize_point_cloud(pcd_with_normals, feature_values, window_name=feature_name)
    
    # Export feature values to CSV
    """
    points = np.asarray(pcd_with_normals.points)
    feature_array = np.column_stack((
        points,
        features['omnivariance'],
        features['eigentropy'],
        features['anisotropy'],
        features['planarity'],
        features['linearity'],
        features['surface_variation'],
        features['sphericity'],
        features['verticality']
    ))
    
    header = "x,y,z,omnivariance,eigentropy,anisotropy,planarity,linearity,surface_variation,sphericity,verticality"
    np.savetxt("geometric_features.csv", feature_array, delimiter=",", header=header)
    """

if __name__ == "__main__":
    main()

# # print("Load a point cloud, print it, and render it")
# print(pcd)

# sample_ply_data = o3d.data.PLYPointCloud()
# pcd = o3d.io.read_point_cloud(sample_ply_data.path)
# print(pcd)
# pcd.rotate(pcd.get_rotation_matrix_from_xyz((0, np.pi, np.pi)))
# o3d.io.write_point_cloud("point_cloud_chair.pcd", pcd)
# o3d.visualization.draw_geometries([pcd])

    # Voxel Grid
# point_cloud_downsampled = pcd.voxel_down_sample(voxel_size=1)
# print("Downsampled point cloud")
# print(point_cloud_downsampled)
# o3d.visualization.draw_geometries([point_cloud_downsampled])
# o3d.io.write_point_cloud("point_cloud_downsampled.pcd", point_cloud_downsampled)

# # Perform plane segmentation using RANSAC
# plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=2000)

# # # # Extract inlier and outlier point clouds
# inlier_cloud = pcd.select_by_index(inliers)
# # outlier_cloud = pcd.select_by_index(inliers, invert=True)
# o3d.visualization.draw_geometries([inlier_cloud])
# o3d.io.write_point_cloud("inlier_cloud.pcd", inlier_cloud)


# pcd_downsampled = o3d.io.read_point_cloud("point_cloud_downsampled.pcd")
# print(pcd_downsampled)

# # inlier_cloud = o3d.io.read_point_cloud("inlier_cloud.pcd")
# # print(inlier_cloud)

# # o3d.visualization.draw_geometries([pcd_downsampled])

# # Estimate normals
# pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
# pcd_downsampled.normalize_normals()

# # You can also orient them consistently (optional)
# pcd_downsampled.orient_normals_consistent_tangent_plane(100)

# # Visualize normals
# o3d.visualization.draw_geometries([pcd_downsampled], point_show_normal=True)

# labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=50, print_progress=True))

# obs = []
# # Group points by cluster label
# indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()

# Iterate over clusters and perform PCA
# for i in range(0, len(indexes)):
#     nb_points = len(pcd.select_by_index(indexes[i]).points)
#     if nb_points > 20 and nb_points < 1000:
#         sub_cloud = pcd.select_by_index(indexes[i])
#         obb = sub_cloud.get_oriented_bounding_box()
#         o3d.visualization.draw_geometries([obb])

