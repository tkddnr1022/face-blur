# Face Blur

폴더 안 사진을 재귀적으로 찾아 YOLO로 얼굴을 탐지한 뒤 블러 처리합니다. 원본 폴더 구조는 그대로 유지됩니다.

## 필요 사항

- Python 3.12 권장 (3.11도 가능)
- 얼굴 전용 YOLO 가중치(`.pt`). `*.pt`는 git에 포함되지 않으므로 별도로 준비하세요.
- GPU 사용 시 NVIDIA 드라이버. CUDA Toolkit을 따로 맞출 필요는 없습니다.

## 설치

가상환경을 만든 뒤 활성화합니다.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
python -m pip install -U pip
```

**torch를 먼저 설치**한 다음 `requirements.txt`를 설치하세요. 순서를 바꾸면 Windows에서 CPU용 torch가 깔릴 수 있습니다.

CPU만 쓰는 경우:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

NVIDIA GPU를 쓰는 경우 (공유 PC에는 `cu130` 권장, 드라이버가 12.6대면 `cu126`):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

설치가 맞는지 확인합니다.

```bash
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available())"
```

- `2.x.x+cpu` / `cuda False` → `--device cuda`를 쓰지 마세요. CPU로 실행하거나 CUDA용 torch를 다시 설치하세요.
- `2.x.x+cu130` / `cuda True` → `--device cuda`를 사용할 수 있습니다.

`nvidia-smi`의 CUDA 버전은 드라이버가 지원하는 최대 버전입니다. PyTorch 휠의 CUDA(예: 13.0)보다 높거나 같으면 됩니다.

## 사용법

```bash
python blur_faces.py --input ./photos --output ./blurred --model yolov12l-face.pt
```

GPU:

```bash
python blur_faces.py --input ./photos --output ./blurred --model yolov12l-face.pt --device cuda
```

얼굴 전용 모델이 없으면, 일반 YOLO 모델로 사람을 잡고 상단 30%를 얼굴로 근사할 수 있습니다.

```bash
python blur_faces.py --input ./photos --output ./blurred --model yolov8n.pt --fallback-person
```

지원 확장자: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`

## 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--input` | 원본 사진 폴더 (하위 폴더 포함) | 필수 |
| `--output` | 결과 저장 폴더 | 필수 |
| `--model` | YOLO 가중치 경로 | 필수 |
| `--conf` | 탐지 신뢰도 임계값 | `0.25` |
| `--blur-strength` | 블러 강도 배수 (클수록 더 흐릿) | `1.0` |
| `--method` | `blur` / `pixelate` / `black` | `blur` |
| `--fallback-person` | person 박스 상단을 얼굴로 간주 | 꺼짐 |
| `--device` | `cpu`, `cuda` 등 | 자동 |
| `--workers` | 병렬 워커 수 (GPU면 1 권장) | `1` |
