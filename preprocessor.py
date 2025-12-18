from __future__ import annotations

from pathlib import Path
from typing import List, Union

import cv2

from filesystem.file_utils import FileUtils

PathLike = Union[str, Path]


class Preprocessor:
    OUTPUT_ROOT = "data_processed"
    SIZE_W = 224
    SIZE_H = 224

    # OpenCV JPG quality: int in [0..100]
    JPEG_QUALITY = 85

    @classmethod
    def go(cls, benign: List[PathLike], malignant: List[PathLike]) -> None:
        cls._process_class(benign, label="benign")
        cls._process_class(malignant, label="malignant")

    @classmethod
    def _process_class(cls, files: List[PathLike], label: str) -> None:
        total = len(files)
        print(f"[Preprocessor] {label}: {total} files -> {cls.OUTPUT_ROOT}/{label}")

        for i, f in enumerate(files):
            src = Path(f)

            ok = cls._process_one(src, out_label=label)
            if (i + 1) % 250 == 0 or (i + 1) == total:
                print(f"  {label}: {i+1}/{total} (last_ok={ok})")

    @classmethod
    def _process_one(cls, src: Path, out_label: str) -> bool:
        if not src.is_file():
            print(f"[WARN] Missing file: {src}")
            return False

        try:
            img = FileUtils.load_opencv_image(src, flags=cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[WARN] Failed to read: {src} ({e})")
            return False

        resized = cv2.resize(img, (cls.SIZE_W, cls.SIZE_H), interpolation=cv2.INTER_AREA)

        out_name = src.stem
        try:
            FileUtils.save_local_opencv_image_jpg(
                resized,
                subdirectory=f"{cls.OUTPUT_ROOT}/{out_label}",
                name=out_name,
                quality=cls.JPEG_QUALITY,   # <-- int 0..100
            )
        except Exception as e:
            print(f"[WARN] Failed to write: {cls.OUTPUT_ROOT}/{out_label}/{out_name}.jpg ({e})")
            return False

        return True
