# Photogrammetry Pipeline – 3D Mesh Generation

Энэхүү проект нь **COLMAP** ашиглан зургуудаас 3D сэргээн босголт (sparse reconstruction) хийж, дараа нь **Poisson surface reconstruction** аргаар mesh үүсгэдэг бүрэн pipeline юм. GPU байгаа тохиолдолд CUDA-г ашиглаж, эс бөгөөс CPU дээр ажиллана.

## Зорилго

Зургийн багцаас:
- Sparse point cloud үүсгэх (`sparse_reconstruction.ply`)
- Дуу чимээг арилгаж цэвэрлэсэн point cloud (`cleaned_pcd.ply`)
- Гөлгөр, бодит mesh файл (`mesh_improved_poisson.obj`)

## Шаардлагатай програм хангамж

- **Python 3.12+**
- **COLMAP** (3.6+ суулгасан байх) – [албан ёсны сайт](https://colmap.github.io/install.html)
- **CUDA** (GPU ашиглах бол) – NVIDIA драйвер, CUDA Toolkit 11.8/12.0
- **Системийн номын сан** (Linux): `libgl1-mesa-glx`, `libglib2.0-0`

## Python номын сангууд

Дараах командыг ажиллуулж хамааралтай багцуудыг суулгана:

```bash
pip install requirement.txt 
```
## Dataset бэлтгэх
### 1. Kaggle dataset авах (жишээ)

Хэрэв та өөрийн зургуудаа ашиглахгүй бол Kaggle-с тохирох dataset татаж аваарай.  
Жишээ нь:
> https://www.kaggle.com/datasets/kpoviesistphane/3d-object-reconstruction-from-images

Dataset-ийг татаж авсны дараа **зургууд нь нэг хавтас** дотор байх ёстой. Дэд хавтасгүй, зөвхөн `.jpg`, `.png`, `.jpeg` файлууд.

### 2. Файлын бүтэц

```
2D_to_3D_object_with_colmap/
├── src/                     # (өгөгдсөн бүх .py файлууд энд)
│   ├── config.py
│   ├── main.py
│   ├── modules/
│   │   ├── cpu_pipeline.py
│   │   ├── gpu_pipeline.py
│   │   ├── mesh.py
│   │   └── point_cloud.py
│   └── utils/
│       ├── colmap_utils.py
│       ├── environment.py
│       └── image_utils.py
├── Datasets/                # (өөрөө үүсгэх)
│   └── Ammonite/            # эсвэл өөрийн dataset нэр
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── images/                  # pipeline өөрөө үүсгэнэ
│   └── Ammonite/
├── project/workspace/       # pipeline өөрөө үүсгэнэ
│   └── Ammonite/
├── requirement.txt   
└── README.md
```

## Тохиргоо – `config.py` файлыг засварлах

`src/config.py` дотор дараах мөрүүдийг өөрийн тохиргоонд тохируулна уу:

```python
# Хүссэн dataset-ийн нэр (жишээ нь "Ammonite", "Snails", "Fossil" гэх мэт)
Dataset = "/Ammonite"

# Зургуудын оролтын зам – өөрийн хавтасны бүтцэд тохируулна
# Жишээ нь: "./Datasets/Ammonite"
IMAGE_INPUT_DIR: str = "../Datasets" + Dataset   # үндсэн project доторх Datasets хавтас

# Бусад замыг ихэвчлэн өөрчлөх шаардлагагүй
# IMAGE_PROCESSED_DIR, WORKSPACE_DIR автомат үүснэ

# Зургийг хэд дахин жижгэрүүлэх (хурд/санах ойд зориулан, 1=анхны хэмжээ)
RESIZE_FACTOR: int = 4
```

**Чухал:**
- Хэрэв таны зургууд өөр газар байрлаж байвал `IMAGE_INPUT_DIR`-ийг шууд зааж өгнө (жишээ нь `"/home/user/MyImages"`).
- `ENVIRONMENT` болон `HAS_GPU` автоматаар илрэх тул өөрчлөхгүй байх.

## Ажиллуулах заавар (Local – CPU/GPU)

1. **COLMAP** суулгасан эсэхээ шалгана:
   ```bash
   colmap -h
   ```
   Хэрэв командын мөрөөс олдохгүй бол PATH-д нэмнэ.

2. Pipeline-г эхлүүлэх:
   ```bash
   cd 2D_to_3D_object_with_colmap/
   python src/main.py
   ```

3. Програм дараах алхмуудыг гүйцэтгэнэ:
    - Зургийг resize хийх (хэрэв `RESIZE_FACTOR > 1`)
    - COLMAP database үүсгэх
    - Feature extraction (GPU/CPU)
    - Feature matching (GPU/CPU)
    - Structure-from-Motion (SfM)
    - Sparse model → PLY экспорт
    - Point cloud цэвэрлэх (статистик, радиус outlier filter)
    - Poisson mesh үүсгэх

4. Гарсан файлууд:
    - `project/workspace/Ammonite/sparse_reconstruction.ply` – анхны sparse point cloud
    - `project/workspace/Ammonite/cleaned_pcd.ply` – цэвэршүүлсэн point cloud
    - `project/workspace/Ammonite/mesh_improved_poisson.obj` – эцсийн mesh

## Pipeline-ийн алхамуудын дэлгэрэнгүй тайлбар

### Зураг боловсруулах (`image_utils.py`)
- Оролтын хавтаснаас зургуудыг уншиж, `IMAGE_PROCESSED_DIR` руу шаардлагатай хэмжээнд resize хийнэ.

### COLMAP CPU/GPU pipeline (`cpu_pipeline.py`, `gpu_pipeline.py`)
- GPU илэрсэн бол `gpu_pipeline` дуудагдаж, амжилтгүй бол `cpu_pipeline` руу бууна.
- SiftExtraction болон SiftMatching-д `use_gpu=1` эсвэл `0` тус тус тохируулагдана.
- Feature extraction дээр хэд хэдэн параметртэй оролдлого хийж (high/medium quality), аль нь амжилттай бол түүнийгээ авна.

### Point cloud цэвэрлэгээ (`point_cloud.py`)
- Давхардсан цэгүүдийг арилгах
- Статистик outlier арилгагч (nb_neighbors=30, std_ratio=2.0)
- Радиус outlier арилгагч (nb_points=12, radius=0.09)
- Нормалыг дахин тооцоолох (Poisson mesh-д зайлшгүй)

### Poisson mesh generation (`mesh.py`)
- Open3D-ийн `create_from_point_cloud_poisson` функц
- Гүйцэтгэлийн гүн (depth=10)
- Нягтын доод хязгаарын хувиар шүүлт (15%)
- Taubin гөлгөржүүлэлт (15 давталт)

## Чанарын шинжилгээ

Pipeline дууссаны дараа консол дээр дараах мэдээлэл хэвлэгдэнэ:
- Хэдэн камерын тохиргоо, зураг, 3D цэг илэрсэн (cameras.txt, images.txt, points3D.txt)
- Цэвэрлэсний дараах цэгийн тоо
- Mesh дахь гурвалжны тоо (Open3D лог)

Жишээ:
```
  Камерын тохиргоо: 1 бичлэг
  Зургийн байрлал (pose): 45 бичлэг
  3D цэгүүд: 18432 бичлэг
```

## Алдаа олж засварлах (Troubleshooting)

### "colmap: command not found"
COLMAP суулгаагүй эсвэл PATH-д байхгүй.
```bash
export PATH=$PATH:/usr/local/bin
```

### GPU илрээгүй ч GPU байгаа
`nvidia-smi` командыг ажиллулаад үзнэ. Хэрэв ажиллахгүй бол драйвер суулгах шаардлагатай.  
Pipeline автоматаар CPU горимд шилжинэ.

### "Database is locked" алдаа
Өмнөх ажиллагаанаас үлдсэн `database.db` файлыг устгаад дахин ажиллуулна:
```bash
rm project/workspace/Ammonite/database.db
```

### Сүлжээний холболт (headless) алдаа
`setup_headless_environment()` функц автоматаар орчны хувьсагчдыг тохируулдаг. Хэрэв `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"` гэх алдаа гарвал:
```bash
export QT_QPA_PLATFORM=offscreen
```

### Хангалтгүй тооны 3D цэг (sparse reconstruction)
- Зургийн тоог нэмэх (дор хаяж 10-20)
- `RESIZE_FACTOR`-ийг багасгах (эсвэл 1 болгох)
- Зургууд дээр хангалттай давхцал (overlap) байгаа эсэхийг шалгах

## Гарцыг харах/ашиглах

- **PLY файлууд** – MeshLab, CloudCompare, Blender-ээр нээж болно.
- **OBJ файл** – Blender, Meshmixer, Unity, Unreal Engine-д импортолж болно.

Хэрэв mesh хэтэрхий "нүхтэй" эсвэл барзгар бол:
- `poisson_depth` параметрийг `mesh.py` дотор нэмэх (жишээ нь 12)
- `density_percentile`-ийг багасгах (жишээ нь 5)
- Цэвэрлэх алхамын `std_ratio`-г чангаруулах