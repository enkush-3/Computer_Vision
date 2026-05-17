"""
CPU ашигласан COLMAP pipeline.
GPU байхгүй үед эсвэл GPU pipeline амжилтгүй болсон үед ашиглана.
"""

import os

from src.utils.colmap_utils import check_directory_structure, run_colmap_command


def run_cpu_feature_extraction(database_path: str, image_dir: str) -> bool:
    """
    CPU ашиглан feature extraction хийнэ (олон fallback тохиргоотой).

    Args:
        database_path: COLMAP database файлын зам.
        image_dir:     Зургийн хавтас.

    Returns:
        Аль нэг тохиргоо амжилттай бол True, бүх амжилтгүй бол False.
    """
    print("\n💻 CPU feature extraction эхэллээ...")

    configs = [
        {
            "cmd": [
                "colmap", "feature_extractor",
                "--database_path",             database_path,
                "--image_path",                image_dir,
                "--ImageReader.single_camera", "1",
                "--SiftExtraction.use_gpu",    "0",
                "--SiftExtraction.max_image_size",   "1000",
                "--SiftExtraction.max_num_features",  "16384",
            ],
            "desc": "CPU SIFT (high quality)",
        },
        {
            "cmd": [
                "colmap", "feature_extractor",
                "--database_path",             database_path,
                "--image_path",                image_dir,
                "--ImageReader.single_camera", "1",
                "--SiftExtraction.use_gpu",    "0",
                "--SiftExtraction.max_image_size",   "600",
                "--SiftExtraction.max_num_features",  "8192",
            ],
            "desc": "CPU SIFT (medium - fallback)",
        },
        {
            "cmd": [
                "colmap", "feature_extractor",
                "--database_path", database_path,
                "--image_path",    image_dir,
                "--SiftExtraction.use_gpu", "0",
            ],
            "desc": "CPU SIFT (minimal - last resort)",
        },
    ]

    for cfg in configs:
        if run_colmap_command(cfg["cmd"], cfg["desc"]):
            return True
        print(f"⚠️ {cfg['desc']} амжилтгүй → дараагийн хувилбар туршиж байна...")

    return False


def run_cpu_pipeline(image_dir: str, workspace_dir: str) -> bool:
    """
    Бүрэн CPU pipeline:
      Database → Feature extraction → Matching → SfM → PLY экспорт.

    Args:
        image_dir:     Боловсруулсан зургийн хавтас.
        workspace_dir: COLMAP workspace хавтас.

    Returns:
        Амжилттай бол True, үгүй бол False.
    """
    print("\n" + "═" * 55)
    print("💻  CPU PIPELINE ЭХЭЛЛЭЭ")
    print("═" * 55)

    database_path     = os.path.join(workspace_dir, "database.db")
    sparse_dir        = os.path.join(workspace_dir, "sparse")
    output_ply        = os.path.join(workspace_dir, "sparse_reconstruction.ply")
    sparse_model_path = os.path.join(sparse_dir, "0")

    if not check_directory_structure(workspace_dir, image_dir):
        return False

    # Алхам 1: Database үүсгэх
    print("\n📋 АЛХАМ 1/5: Database үүсгэх")
    if not run_colmap_command(
        ["colmap", "database_creator", "--database_path", database_path],
        "Database creator",
    ):
        return False

    # Алхам 2: Feature extraction (CPU)
    print("\n📋 АЛХАМ 2/5: Feature extraction (CPU)")
    if not run_cpu_feature_extraction(database_path, image_dir):
        print("❌ Бүх feature extraction хувилбар амжилтгүй")
        return False

    # Алхам 3: Feature matching (CPU)
    print("\n📋 АЛХАМ 3/5: Feature matching (CPU)")
    if not run_colmap_command(
        [
            "colmap", "exhaustive_matcher",
            "--database_path",                database_path,
            "--SiftMatching.guided_matching", "1",
            "--SiftMatching.use_gpu",         "0",
            "--SiftMatching.max_ratio",       "0.75",
            "--SiftMatching.max_distance",    "0.7",
        ],
        "CPU exhaustive matching",
    ):
        return False

    # Алхам 4: Structure-from-Motion
    print("\n📋 АЛХАМ 4/5: Structure-from-Motion (mapper)")
    os.makedirs(sparse_dir, exist_ok=True)
    if not run_colmap_command(
        [
            "colmap", "mapper",
            "--database_path",                    database_path,
            "--image_path",                       image_dir,
            "--output_path",                      sparse_dir,
            "--Mapper.ba_refine_focal_length",    "0",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.init_min_tri_angle",        "4",
            "--Mapper.multiple_models",           "0",
            "--Mapper.extract_colors",            "1",
        ],
        "Structure-from-Motion (SfM)",
    ):
        return False

    if not os.path.exists(sparse_model_path):
        print("❌ Sparse model үүсгэгдсэнгүй")
        return False

    # 3D цэгийн тоог харуулах
    points_file = os.path.join(sparse_model_path, "points3D.txt")
    if os.path.exists(points_file):
        with open(points_file, "r") as f:
            lines = [ln for ln in f if not ln.startswith("#") and ln.strip()]
        print(f"📊 Sparse reconstruction: {len(lines)} 3D цэг")

    # Алхам 5: PLY экспорт
    print("\n📋 АЛХАМ 5/5: Sparse model → PLY хөрвүүлэх")
    if not run_colmap_command(
        [
            "colmap", "model_converter",
            "--input_path",  sparse_model_path,
            "--output_path", output_ply,
            "--output_type", "PLY",
        ],
        "PLY экспорт",
    ):
        return False

    # Нэмэлт TXT экспорт
    run_colmap_command(
        [
            "colmap", "model_converter",
            "--input_path",  sparse_model_path,
            "--output_path", workspace_dir,
            "--output_type", "TXT",
        ],
        "TXT экспорт (нэмэлт)",
    )

    return _check_ply_output(output_ply, "CPU")


def analyze_reconstruction_quality(workspace_dir: str) -> None:
    """Sparse reconstruction-ын чанарыг шинжилнэ."""
    print("\n📊 Reconstruction чанарыг шинжилж байна...")
    sparse_model_path = os.path.join(workspace_dir, "sparse", "0")

    for filename, description in [
        ("cameras.txt",  "Камерын тохиргоо"),
        ("images.txt",   "Зургийн байрлал (pose)"),
        ("points3D.txt", "3D цэгүүд"),
    ]:
        filepath = os.path.join(sparse_model_path, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                lines = [ln for ln in f if not ln.startswith("#") and ln.strip()]
            print(f"  ✅ {description}: {len(lines)} бичлэг")
        else:
            print(f"  ❌ {description}: файл олдсонгүй")


def _check_ply_output(output_ply: str, mode: str) -> bool:
    """PLY файл амжилттай үүссэн эсэхийг шалгана."""
    if os.path.exists(output_ply):
        size = os.path.getsize(output_ply)
        print(f"\n🎉 {mode} pipeline амжилттай дууслаа!")
        print(f"✅ PLY файл: {output_ply}  ({size/1024:.1f} KB)")
        try:
            with open(output_ply, "r", errors="ignore") as f:
                for line in f.read(1000).split("\n"):
                    if "element vertex" in line:
                        print(f"📊 Vertices: {line.split()[-1]}")
        except Exception:
            pass
        return True
    else:
        print(f"❌ PLY файл үүсгэгдсэнгүй: {output_ply}")
        return False
