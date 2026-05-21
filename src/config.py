from src.utils.environment import  detect_gpu
import os

HAS_GPU: bool    = detect_gpu()

Dataset = "/Ammonite"

IMAGE_INPUT_DIR: str = "../Datasets" + Dataset
IMAGE_PROCESSED_DIR: str = "../images" + Dataset
WORKSPACE_DIR: str       = "../project/workspace" + Dataset

RESIZE_FACTOR: int = 4

SPARSE_PLY_PATH:   str = os.path.join(WORKSPACE_DIR, "sparse_reconstruction.ply")
CLEANED_PLY_PATH:  str = os.path.join(WORKSPACE_DIR, "cleaned_pcd.ply")
OUTPUT_MESH_PATH:  str = os.path.join(WORKSPACE_DIR, "mesh_improved_poisson.obj")
