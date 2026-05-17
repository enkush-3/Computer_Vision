"""
Тохиргооны файл — замууд болон параметрүүдийг эндээс өөрчилнө.
"""

from src.utils.environment import detect_environment, detect_gpu

# ─────────────────────────────────────────────────────────────
# Орчин болон GPU автоматаар тодорхойлогдоно
# ─────────────────────────────────────────────────────────────
ENVIRONMENT: str = detect_environment()
HAS_GPU: bool    = detect_gpu()

# ─────────────────────────────────────────────────────────────
# Замын тохиргоо
# ─────────────────────────────────────────────────────────────

Dataset = "/Ammonite"

if ENVIRONMENT == "colab":
    IMAGE_INPUT_DIR: str = "/content/drive/MyDrive/Colab Notebooks/Biy_Daalt/Snails/"
else:
    IMAGE_INPUT_DIR: str = "../Datasets" + Dataset

IMAGE_PROCESSED_DIR: str = "../images" + Dataset
WORKSPACE_DIR: str       = "../project/workspace" + Dataset

# ─────────────────────────────────────────────────────────────
# Боловсруулалтын параметрүүд
# ─────────────────────────────────────────────────────────────
RESIZE_FACTOR: int = 4   # 1=original, 2=half, 4=quarter

# ─────────────────────────────────────────────────────────────
# Гаралтын замууд (workspace-аас автоматаар тооцно)
# ─────────────────────────────────────────────────────────────
import os

SPARSE_PLY_PATH:   str = os.path.join(WORKSPACE_DIR, "sparse_reconstruction.ply")
CLEANED_PLY_PATH:  str = os.path.join(WORKSPACE_DIR, "cleaned_pcd.ply")
OUTPUT_MESH_PATH:  str = os.path.join(WORKSPACE_DIR, "mesh_improved_poisson.obj")
