"""
Point cloud боловсруулах: ачаалах, цэвэрлэх, ML denoising.
"""

import os

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from pyntcloud import PyntCloud


# ─────────────────────────────────────────────────────────────
# Point Cloud ачаалах
# ─────────────────────────────────────────────────────────────

def load_point_cloud(ply_path: str) -> "PyntCloud":
    """
    PLY файлыг ачаалж, медиан офсетоор нормчилно.

    Args:
        ply_path: PLY файлын зам.

    Returns:
        Нормчилогдсон PyntCloud объект.

    Raises:
        FileNotFoundError: PLY файл олдоогүй үед.
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(
            f"❌ PLY файл олдсонгүй: {ply_path}\n"
            f"   Дээрх pipeline амжилттай дуусаагүй байж болно."
        )

    cloud = PyntCloud.from_file(ply_path)
    print(cloud)

    df = cloud.points
    offset = df[["x", "y", "z"]].median()
    df[["x", "y", "z"]] -= offset

    print(f"\n📊 Цэгийн тоо: {len(df)}")
    print(f"📊 Баганууд  : {df.columns.tolist()}")
    return cloud


# ─────────────────────────────────────────────────────────────
# Algorithmic цэвэрлэгээ
# ─────────────────────────────────────────────────────────────

def deep_clean_pcd(path: str, output_path: str | None = None) -> o3d.geometry.PointCloud:
    """
    Open3D ашиглан point cloud-г цэвэрлэнэ:
      1. Давхардсан цэг арилгах
      2. Statistical Outlier Removal (SOR)
      3. Radius Outlier Removal (ROR)
      4. Normal тооцоолох

    Args:
        path:        Оролтын PLY файлын зам.
        output_path: Хадгалах зам (None бол хадгалахгүй).

    Returns:
        Цэвэрлэгдсэн PointCloud объект.
    """
    pcd = o3d.io.read_point_cloud(path)
    print(f"Эхний цэгүүд: {len(pcd.points)}")

    # 1. Давхардсан цэг
    pcd = pcd.remove_duplicated_points()

    # 2. Statistical Outlier Removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)

    # 3. Radius Outlier Removal
    cl, ind = pcd.remove_radius_outlier(nb_points=12, radius=0.09)
    pcd = pcd.select_by_index(ind)

    # 4. Normal тооцоолох (Poisson mesh-д шаардлагатай)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    print(f"Цэвэрлэсний дараах цэгүүд: {len(pcd.points)}")

    if output_path:
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"✅ Хадгалагдлаа: {output_path}")

    return pcd