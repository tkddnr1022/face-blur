"""
YOLO를 이용한 대량 사진 얼굴 블러 처리 스크립트
------------------------------------------------
사용 예:
    python blur_faces.py --input ./photos --output ./blurred --model yolov8n-face.pt
    python blur_faces.py --input ./photos --output ./blurred --model yolov8n.pt --fallback-person

옵션:
    --input           원본 사진 폴더 (하위 폴더까지 재귀 탐색)
    --output          결과 저장 폴더 (원본 폴더 구조 유지)
    --model           YOLO 가중치 경로 (.pt). 얼굴 전용 모델 권장
    --conf            탐지 신뢰도 임계값 (기본 0.25)
    --blur-strength   블러 커널 크기 배수 (기본 1.0, 크게 할수록 더 흐릿)
    --method          blur | pixelate | black  (블러 방식 선택)
    --fallback-person 얼굴 전용 모델이 없을 때, person 박스 상단 30%를 얼굴로 간주
    --device          cpu | cuda:0 등 (기본: 자동)
    --workers         이미지 로딩용 스레드 수 (기본 4)
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def apply_blur(img, box, method="blur", strength=1.0):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return img

    roi = img[y1:y2, x1:x2]

    if method == "black":
        img[y1:y2, x1:x2] = 0
        return img

    if method == "pixelate":
        # 축소 후 확대하여 모자이크 효과
        scale = max(1, int(min(x2 - x1, y2 - y1) / (10 * strength)))
        small = cv2.resize(roi, (max(1, (x2 - x1) // scale), max(1, (y2 - y1) // scale)),
                            interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        img[y1:y2, x1:x2] = pixelated
        return img

    # 기본: 가우시안 블러
    k = int(max(x2 - x1, y2 - y1) * 0.6 * strength)
    k = k if k % 2 == 1 else k + 1
    k = max(k, 5)
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    img[y1:y2, x1:x2] = blurred
    return img


def find_images(input_dir: Path):
    return [p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]


def process_image(path: Path, input_dir: Path, output_dir: Path, model, args):
    img = cv2.imread(str(path))
    if img is None:
        print(f"[경고] 읽기 실패, 건너뜀: {path}")
        return False

    results = model.predict(
        source=img,
        conf=args.conf,
        device=args.device,
        verbose=False,
    )

    boxes_found = 0
    for r in results:
        if r.boxes is None:
            continue
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = names.get(cls_id, "") if isinstance(names, dict) else names[cls_id]

            xyxy = box.xyxy[0].tolist()

            if args.fallback_person:
                # person 클래스만 사용, 상단 30%를 얼굴 영역으로 근사
                if label != "person":
                    continue
                x1, y1, x2, y2 = xyxy
                face_h = (y2 - y1) * 0.3
                xyxy = [x1, y1, x2, y1 + face_h]
            # 얼굴 전용 모델이면 클래스 상관없이 모든 박스를 얼굴로 처리

            img = apply_blur(img, xyxy, method=args.method, strength=args.blur_strength)
            boxes_found += 1

    rel_path = path.relative_to(input_dir)
    save_path = output_dir / rel_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)
    return boxes_found > 0


def main():
    parser = argparse.ArgumentParser(description="YOLO 기반 대량 얼굴 블러 처리")
    parser.add_argument("--input", required=True, help="원본 사진 폴더")
    parser.add_argument("--output", required=True, help="결과 저장 폴더")
    parser.add_argument("--model", required=True, help="YOLO 가중치 경로")
    parser.add_argument("--conf", type=float, default=0.25, help="탐지 신뢰도 임계값")
    parser.add_argument("--blur-strength", type=float, default=1.0, help="블러 강도 배수")
    parser.add_argument("--method", choices=["blur", "pixelate", "black"], default="blur")
    parser.add_argument("--fallback-person", action="store_true",
                         help="얼굴 전용 모델이 없을 때 person 박스 상단부를 얼굴로 간주")
    parser.add_argument("--device", default=None, help="cpu 또는 cuda:0 등 (기본: 자동감지)")
    parser.add_argument("--workers", type=int, default=1,
                         help="병렬 처리 워커 수 (GPU 사용 시 1 권장, CPU는 늘려도 됨)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        print(f"입력 폴더가 존재하지 않습니다: {input_dir}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"모델 로드 중: {args.model}")
    model = YOLO(args.model)

    images = find_images(input_dir)
    print(f"총 {len(images)}장의 이미지를 찾았습니다.")
    if not images:
        return

    processed = 0
    faces_total = 0

    def _job(p):
        return process_image(p, input_dir, output_dir, model, args)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, found in enumerate(ex.map(_job, images), 1):
                processed += 1
                faces_total += int(found)
                if i % 50 == 0 or i == len(images):
                    print(f"진행: {i}/{len(images)}")
    else:
        for i, p in enumerate(images, 1):
            found = _job(p)
            processed += 1
            faces_total += int(found)
            if i % 50 == 0 or i == len(images):
                print(f"진행: {i}/{len(images)}")

    print(f"완료: {processed}장 처리, {faces_total}장에서 얼굴 탐지됨")
    print(f"결과 저장 위치: {output_dir.resolve()}")


if __name__ == "__main__":
    main()