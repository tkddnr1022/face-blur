# Face Blur

폴더 안 사진을 재귀적으로 찾아 YOLO로 얼굴을 탐지한 뒤 블러 처리합니다. 원본 폴더 구조는 그대로 유지됩니다.

## 필요 사항

가상환경 생성 후 활성화합니다.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
```

```bash
pip install ultralytics opencv-python numpy
```

얼굴 전용 YOLO 가중치(`.pt`)를 준비하세요. 기본값은 `yolo26n-face.pt`입니다.

## 사용법

```bash
python blur_faces.py --input ./photos --output ./blurred --model yolo26n-face.pt --blur-strength 2 --conf 0.5
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
| `--model` | YOLO 가중치 경로 | `yolo26n-face.pt` |
| `--conf` | 탐지 신뢰도 임계값 | `0.25` |
| `--blur-strength` | 블러 강도 배수 (클수록 더 흐릿) | `1.0` |
| `--method` | `blur` / `pixelate` / `black` | `blur` |
| `--fallback-person` | person 박스 상단을 얼굴로 간주 | 꺼짐 |
| `--device` | `cpu`, `cuda:0` 등 | 자동 |
| `--workers` | 병렬 워커 수 (GPU면 1 권장) | `1` |
