"""
Mesh үүсгэх: Poisson surface reconstruction.
"""

import numpy as np
import open3d as o3d


def build_poisson_mesh(
    input_ply: str,
    output_obj: str,
    poisson_depth: int = 10,
    density_percentile: float = 15.0,
    smooth_iterations: int = 15,
) -> o3d.geometry.TriangleMesh:
    """
    ML-ээр цэвэрлэгдсэн point cloud-аас Poisson mesh үүсгэнэ.

    Args:
        input_ply:          Оролтын PLY файлын зам.
        output_obj:         Гаралтын OBJ файлын зам.
        poisson_depth:      Poisson reconstruction depth (өндөр = нарийн, удаан).
        density_percentile: Нягтрал доогуур цэгүүдийг арилгах хувь.
        smooth_iterations:  Taubin smoothing давталтын тоо.

    Returns:
        Үүсгэгдсэн TriangleMesh объект.
    """
    print("🚀 Сайжруулсан Algorithmic Mesh эхэллээ...")

    pcd = o3d.io.read_point_cloud(input_ply)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=50)
    )
    pcd.orient_normals_consistent_tangent_plane(k=20)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth, width=0, scale=1.2, linear_fit=True
        )

    # Нягтрал доогуур vertex-үүдийг арилгах
    densities_np = np.asarray(densities)
    threshold = np.percentile(densities_np, density_percentile)
    mesh.remove_vertices_by_mask(densities_np < threshold)

    # Smoothing + normal тооцоолох
    mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(output_obj, mesh)
    print(f"✅ Сайжруулсан Poisson mesh хадгалагдлаа: {output_obj}")
    return mesh
