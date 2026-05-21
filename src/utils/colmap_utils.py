import os
import subprocess
import time
from typing import List


def run_colmap_command(
    command: List[str],
    description: str,
    timeout: int = 3600,
    verbose: bool = True,
) -> bool:

    print(f"\n{'─'*55}")
    print(f"  {description}")
    print(f"  Команд: {' '.join(command)}")
    print(f"{'─'*55}")
    t_start = time.time()

    try:
        env = os.environ.copy()

        if verbose:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
            )
            deadline = time.time() + timeout

            for line in process.stdout:
                line = line.rstrip()
                if line:
                    print(f"  {line}")
                if time.time() > deadline:
                    process.kill()
                    print(f"\n{description} - хугацаа хэтэрлээ ({timeout}с)")
                    return False

            process.wait()
            returncode = process.returncode
        else:
            result = subprocess.run(
                command, capture_output=True, text=True,
                timeout=timeout, env=env
            )
            returncode = result.returncode
            if result.stdout:
                print(result.stdout[-1000:])
            if result.stderr and returncode != 0:
                print(result.stderr[-500:])

        elapsed = time.time() - t_start
        if returncode == 0:
            print(f"\n{description} амжилттай дууслаа ({elapsed:.1f}с)")
            return True
        else:
            print(f"\n{description} амжилтгүй (return code: {returncode}, {elapsed:.1f}с)")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n{description} - хугацаа хэтэрлээ ({timeout}с)")
        return False
    except Exception as e:
        print(f"\n{description} - алдаа гарлаа: {e}")
        return False


def check_directory_structure(workspace_dir: str, image_dir: str) -> bool:
    print(f"\nХавтасны бүтцийг шалгаж байна...")

    if not os.path.exists(image_dir):
        print(f"Зургийн хавтас олдсонгүй: {image_dir}")
        return False

    imgs = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
    ]
    if len(imgs) < 2:
        print(f"Зураг хангалтгүй ({len(imgs)} ширхэг). Дор хаяж 2 шаардлагатай.")
        return False

    print(f"{len(imgs)} зураг олдлоо → {image_dir}/")
    os.makedirs(workspace_dir, exist_ok=True)
    print(f"Workspace хавтас бэлэн: {workspace_dir}/")
    return True
