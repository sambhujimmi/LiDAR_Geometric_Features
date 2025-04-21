# Geometric Feature Extraction from LiDAR Point Cloud

Photogrammetry course project

## Notes

- Python 3.11 for Open3D

```
/opt/homebrew/bin/python3.11 -m venv venv
```

## Steps

- Load point cloud

```
pcd = o3d.io.read_point_cloud(pcd_path)
```

- Voxel downsampling

```
point_cloud_downsampled = pcd.voxel_down_sample(voxel_size=1)
```

- Remove outliers

- Compute surface normals

- PCA and Eigen values

- Estimate geometric features

- Visualization
