"""
Орчин болон GPU тодорхойлох хэрэгслүүд.
"""

import subprocess


def detect_environment() -> str:
    """Google Colab эсвэл local Jupyter орчинг тодорхойлно."""
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        return "local"


def detect_gpu() -> bool:
    """GPU (CUDA) байгаа эсэхийг шалгана. Байвал True, байхгүй бол False."""
    print("🔍 GPU шалгаж байна...")
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpu_name = "Unknown GPU"
            for line in result.stdout.split("\n"):
                for keyword in ["Tesla", "RTX", "GTX", "A100", "V100", "T4", "P100", "A10", "L4", "MX"]:
                    if keyword in line:
                        gpu_name = line.strip()
                        break
            print(f"✅ GPU олдлоо: {gpu_name}")
            for line in result.stdout.split("\n")[:15]:
                if line.strip():
                    print(f"   {line}")
            return True
        else:
            print("❌ nvidia-smi амжилтгүй → GPU байхгүй")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi олдсонгүй → NVIDIA драйвер суугаагүй")
        return False
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi хугацаа хэтэрлээ → GPU гэж үзэхгүй")
        return False
    except Exception as e:
        print(f"❌ GPU шалгахад алдаа гарлаа: {e}")
        return False
