import os
import time

from src import config
from src.modules.cpu_pipeline import analyze_reconstruction_quality, run_cpu_pipeline
from src.modules.gpu_pipeline import run_gpu_pipeline
from src.modules.mesh import build_poisson_mesh
from src.modules.point_cloud import deep_clean_pcd
from src.utils.image_utils import preprocess_images


def setup_headless_environment() -> None:
    env_vars = {
        "QT_QPA_PLATFORM":           "offscreen",
        "DISPLAY":                   "",
        "QT_QPA_FONTDIR":            "/usr/share/fonts",
        "MESA_GL_VERSION_OVERRIDE":  "3.3",
        "MESA_GLSL_VERSION_OVERRIDE": "330",
    }
    for key, value in env_vars.items():
        os.environ[key] = value
    print("Headless орчин тохируулагдлаа")


def setup_cuda_environment(has_gpu: bool) -> bool:
    if not has_gpu:
        return False

    print("CUDA орчин тохируулж байна...")
    cuda_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/local/cuda-12.0",
        "/usr/local/cuda-11.8",
    ]
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"CUDA олдлоо: {path}")
            os.environ["CUDA_HOME"]  = path
            os.environ["CUDA_ROOT"]  = path
            cuda_bin = os.path.join(path, "bin")
            cuda_lib = os.path.join(path, "lib64")
            if cuda_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{cuda_bin}:{os.environ.get('PATH', '')}"
            cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
            if cuda_lib not in cur_ld:
                os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib}:{cur_ld}"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            return True

    print("CUDA суулгацын зам олдсонгүй")
    return False


def print_summary(has_gpu: bool) -> None:
    print()
    print("═" * 55)
    print("    ТОХИРГООНЫ ХУРААНГУЙ")
    print("═" * 55)
    print(f"   Хурдасгуур: {'GPU (CUDA)' if has_gpu else 'CPU (GPU байхгүй)'}")
    print(f"   Зураг орол.: {config.IMAGE_INPUT_DIR}")
    print(f"   Workspace  : {config.WORKSPACE_DIR}")
    print("═" * 55)


def run_colmap_stage(has_gpu: bool) -> bool:
    setup_headless_environment()

    if has_gpu:
        print("\nGPU олдсон → GPU pipeline-г эхлүүлж байна...")
        setup_cuda_environment(has_gpu)
        success = run_gpu_pipeline(config.IMAGE_PROCESSED_DIR, config.WORKSPACE_DIR)
        if not success:
            print("\nGPU pipeline амжилтгүй → CPU pipeline рүү шилжиж байна...")
            success = run_cpu_pipeline(config.IMAGE_PROCESSED_DIR, config.WORKSPACE_DIR)
    else:
        print("\nGPU байхгүй → CPU pipeline-г эхлүүлж байна...")
        success = run_cpu_pipeline(config.IMAGE_PROCESSED_DIR, config.WORKSPACE_DIR)

    return success


def main() -> bool:
    t0 = time.time()
    has_gpu     = config.HAS_GPU

    print("\n" + "═" * 55)
    print("PHOTOGRAMMETRY PIPELINE ЭХЭЛЛЭЭ")
    print("═" * 55)
    print_summary(has_gpu)

    print("\nАЛХАМ 1: Зургуудыг боловсруулах")
    try:
        preprocess_images(
            input_dir=config.IMAGE_INPUT_DIR,
            output_dir=config.IMAGE_PROCESSED_DIR,
            resize_factor=config.RESIZE_FACTOR,
        )
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return False

    print("\nАЛХАМ 2: COLMAP pipeline")
    if not run_colmap_stage(has_gpu):
        print("\nCOLMAP PIPELINE АМЖИЛТГҮЙ ДУУСЛАА")
        return False

    print("\nАЛХАМ 3: Point cloud цэвэрлэгээ")
    deep_clean_pcd(
        path=config.SPARSE_PLY_PATH,
        output_path=config.CLEANED_PLY_PATH,
    )

    print("\nАЛХАМ 4: Poisson mesh үүсгэх")
    build_poisson_mesh(
        input_ply=config.CLEANED_PLY_PATH,
        output_obj=config.OUTPUT_MESH_PATH,
    )

    total_time = time.time() - t0
    print("\n" + "═" * 55)
    analyze_reconstruction_quality(config.WORKSPACE_DIR)
    print("\n  PIPELINE АМЖИЛТТАЙ ДУУСЛАА!")
    print(f"  Нийт хугацаа: {total_time / 60:.1f} минут")
    print("\n Дараагийн алхамууд:")

    print("   1. PLY файлыг татаж авна")
    print("   2. MeshLab эсвэл CloudCompare-р дүрслэнэ")
    print("   3. Шаардлагатай бол mesh generation хийнэ")
    print("═" * 55)
    return True

if __name__ == "__main__":
    success = main()
    print(f"\n{' Pipeline амжилттай дууслаа!' if success else '❌ Pipeline амжилтгүй.'}")