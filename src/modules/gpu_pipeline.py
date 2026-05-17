"""
GPU ашигласан COLMAP pipeline.
GPU байвал ашиглана; амжилтгүй бол CPU pipeline руу буцна.
"""

import os

from src.utils.colmap_utils import check_directory_structure, run_colmap_command


def run_gpu_feature_extraction(database_path: str, image_dir: str) -> bool:
    """
    GPU ашиглан feature extraction хийнэ.
    Амжилтгүй бол дараагийн (бага чанарын) тохиргоо руу буцна.

    Args:
        database_path: COLMAP database файлын зам.
        image_dir:     Зургийн хавтас.

    Returns:
        Аль нэг тохиргоо амжилттай бол True, бүх амжилтгүй бол False.
    """
    print("\n🎮 GPU feature extraction эхэллээ...")

    configs = [
        {
            "cmd": [
                "colmap", "feature_extractor",
                "--database_path",             database_path,
                "--image_path",                image_dir,
                "--ImageReader.single_camera", "1",
                "--SiftExtraction.use_gpu",    "1",
                "--SiftExtraction.gpu_index",  "0",
                "--SiftExtraction.max_image_size",   "3200",
                "--SiftExtraction.max_num_features",  "16384",
            ],
            "desc": "GPU SIFT (high quality)",
        },
        {
            "cmd": [
                "colmap", "feature_extractor",
                "--database_path",             database_path,
                "--image_path",                image_dir,
                "--ImageReader.single_camera", "1",
                "--SiftExtraction.use_gpu",    "1",
                "--SiftExtraction.gpu_index",  "0",
                "--SiftExtraction.max_image_size",   "1600",
                "--SiftExtraction.max_num_features",  "8192",
            ],
            "desc": "GPU SIFT (medium - fallback)",
        },
    ]

    for cfg in configs:
        if run_colmap_command(cfg["cmd"], cfg["desc"]):
            return True
        print(f"⚠️ {cfg['desc']} амжилтгүй → дараагийн хувилбар туршиж байна...")

    print("⚠️ GPU feature extraction амжилтгүй → CPU рүү шилжиж байна")
    return False


def run_gpu_pipeline(image_dir: str, workspace_dir: str) -> bool:
    """
    Бүрэн GPU pipeline:
      Database → Feature extraction → Matching → SfM → PLY экспорт.

    Args:
        image_dir:     Боловсруулсан зургийн хавтас.
        workspace_dir: COLMAP workspace хавтас.

    Returns:
        Амжилттай бол True, үгүй бол False.
    """
    print("\n" + "═" * 55)
    print("🚀  GPU PIPELINE ЭХЭЛЛЭЭ")
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

    # Алхам 2: Feature extraction (GPU → CPU fallback)
    print("\n📋 АЛХАМ 2/5: Feature extraction (GPU)")
    if not run_gpu_feature_extraction(database_path, image_dir):
        print("⚠️ GPU feature extraction бүрэн амжилтгүй → CPU ашиглана")
        from src.modules.cpu_pipeline import run_cpu_feature_extraction
        if not run_cpu_feature_extraction(database_path, image_dir):
            return False

    # Алхам 3: Feature matching (GPU)
    print("\n📋 АЛХАМ 3/5: Feature matching (GPU)")
    matched = run_colmap_command(
        [
            "colmap", "exhaustive_matcher",
            "--database_path",                database_path,
            "--SiftMatching.use_gpu",         "1",
            "--SiftMatching.gpu_index",       "0",
            "--SiftMatching.guided_matching",  "1",
            "--SiftMatching.max_ratio",        "0.75",
            "--SiftMatching.max_distance",     "0.7",
        ],
        "GPU exhaustive matching",
    )

    if not matched:
        print("⚠️ GPU matching амжилтгүй → CPU matching туршиж байна")
        if not run_colmap_command(
            [
                "colmap", "exhaustive_matcher",
                "--database_path",                database_path,
                "--SiftMatching.use_gpu",         "0",
                "--SiftMatching.guided_matching",  "1",
                "--SiftMatching.max_ratio",        "0.75",
                "--SiftMatching.max_distance",     "0.7",
            ],
            "CPU exhaustive matching (fallback)",
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

    return _check_ply_output(output_ply, "GPU")


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
