# file_utils.py

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any, Union
from pathlib import Path
from io import BytesIO

from filesystem.file_io import FileIO

PathLike = Union[str, Path]

if TYPE_CHECKING:
    import numpy as np


class FileUtils:
    """
    High-level convenience wrappers around FileIO for common content types:
      - text
      - json (lazy import)
      - images: Pillow (lazy), Bitmap (lazy), OpenCV/numpy (lazy)

    Hard rule:
      - FileUtils never mkdirs. All directory creation must happen inside FileIO.save().
      - Therefore, all image saves are: encode -> bytes -> FileIO.save(bytes, path).

    OpenCV flags quick guide (cv2.IMREAD_*):
      - 1  IMREAD_COLOR: BGR (3 channels), drops alpha if present (default)
      - 0  IMREAD_GRAYSCALE: single-channel grayscale
      - -1 IMREAD_UNCHANGED: keep original channels (e.g., BGRA if PNG has alpha)

    Channel notes:
      - OpenCV color images are BGR order by default.
      - If you need RGB (typical for ML), convert:
          rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    """

    # ================================================================
    # TEXT UTILITIES
    # ================================================================

    @classmethod
    def load_text(cls, file_path: PathLike, encoding: str = "utf-8") -> str:
        return FileIO.load(file_path).decode(encoding)

    @classmethod
    def save_text(cls, text: str, file_path: PathLike, encoding: str = "utf-8") -> Path:
        return FileIO.save(text.encode(encoding), file_path)

    @classmethod
    def load_local_text(cls, subdirectory: Optional[str], name: str, extension: str, encoding: str = "utf-8") -> str:
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.load_text(path, encoding)

    @classmethod
    def save_local_text(cls, text: str, subdirectory: Optional[str], name: str, extension: str, encoding: str = "utf-8") -> Path:
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.save_text(text, path, encoding)

    # ================================================================
    # JSON UTILITIES (lazy import)
    # ================================================================

    @classmethod
    def load_json(cls, file_path: PathLike, encoding: str = "utf-8") -> Any:
        import json  # lazy
        return json.loads(cls.load_text(file_path, encoding=encoding))

    @classmethod
    def save_json(
        cls,
        obj: Any,
        file_path: PathLike,
        encoding: str = "utf-8",
        indent: int = 2,
        sort_keys: bool = False,
    ) -> Path:
        import json  # lazy
        text = json.dumps(obj, indent=indent, sort_keys=sort_keys)
        return cls.save_text(text, file_path, encoding=encoding)

    @classmethod
    def load_local_json(cls, subdirectory: Optional[str], name: str, extension: str = "json", encoding: str = "utf-8") -> Any:
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.load_json(path, encoding=encoding)

    @classmethod
    def save_local_json(
        cls,
        obj: Any,
        subdirectory: Optional[str],
        name: str,
        extension: str = "json",
        encoding: str = "utf-8",
        indent: int = 2,
        sort_keys: bool = False,
    ) -> Path:
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.save_json(obj, path, encoding=encoding, indent=indent, sort_keys=sort_keys)

    # ================================================================
    # PILLOW LOAD/SAVE (lazy import)
    # ================================================================

    @classmethod
    def load_pillow_image(cls, file_path: PathLike) -> Any:
        from PIL import Image  # lazy

        path = Path(file_path).resolve()
        if path.is_file():
            im = Image.open(path)
            im.load()
            return im

        base = path.with_suffix("")
        for ext in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".tif", ".tiff"]:
            attempt = base.with_suffix(ext)
            if attempt.is_file():
                im = Image.open(attempt)
                im.load()
                return im

        raise FileNotFoundError(f"Pillow image not found: {path}")

    @classmethod
    def save_pillow_image_png(cls, image: Any, file_path: PathLike) -> Path:
        bio = BytesIO()
        image.save(bio, format="PNG")
        return FileIO.save(bio.getvalue(), file_path)

    @classmethod
    def save_pillow_image_jpg(cls, image: Any, file_path: PathLike, quality: int = 95) -> Path:
        """
        Save a Pillow Image as JPG.

        quality: int in [1..95] (common useful JPEG quality range for Pillow)
        - We enforce the range strictly and raise if out of range.
        - JPEG does not support alpha; images with alpha are converted to RGB.
        """
        if not isinstance(quality, int):
            raise TypeError("Pillow JPG quality must be an int in [1..95].")
        if quality < 1 or quality > 95:
            raise ValueError(f"Pillow JPG quality out of range: {quality} (expected 1..95).")

        im = image
        mode = getattr(im, "mode", None)
        if mode in ("RGBA", "LA"):
            im = im.convert("RGB")
        elif mode not in ("RGB", "L"):
            im = im.convert("RGB")

        bio = BytesIO()
        im.save(bio, format="JPEG", quality=quality, optimize=True)
        return FileIO.save(bio.getvalue(), file_path)

    @classmethod
    def load_local_pillow_image(cls, subdirectory: Optional[str], name: str, extension: Optional[str] = None) -> Any:
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.load_pillow_image(path)

    @classmethod
    def save_local_pillow_image_png(cls, image: Any, subdirectory: Optional[str], name: str) -> Path:
        path = FileIO.local_file(subdirectory, name, "png")
        return cls.save_pillow_image_png(image, path)

    @classmethod
    def save_local_pillow_image_jpg(cls, image: Any, subdirectory: Optional[str], name: str, quality: int = 95) -> Path:
        path = FileIO.local_file(subdirectory, name, "jpg")
        return cls.save_pillow_image_jpg(image, path, quality=quality)

    # ================================================================
    # BITMAP LOAD/SAVE (lazy import)
    # ================================================================

    @classmethod
    def load_bitmap(cls, file_path: PathLike) -> Any:
        from image.bitmap import Bitmap  # lazy

        im = cls.load_pillow_image(file_path)
        bmp = Bitmap()
        bmp.import_pillow(im)
        return bmp

    @classmethod
    def save_bitmap_png(cls, bitmap: Any, file_path: PathLike) -> Path:
        im = bitmap.export_pillow()
        return cls.save_pillow_image_png(im, file_path)

    @classmethod
    def save_bitmap_jpg(cls, bitmap: Any, file_path: PathLike, quality: int = 95) -> Path:
        """
        quality: int in [1..95] (Pillow JPEG quality range we enforce)
        """
        im = bitmap.export_pillow()
        return cls.save_pillow_image_jpg(im, file_path, quality=quality)

    @classmethod
    def load_local_bitmap(cls, subdirectory: Optional[str], name: str, extension: Optional[str] = None) -> Any:
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.load_bitmap(path)

    @classmethod
    def save_local_bitmap_png(cls, bitmap: Any, subdirectory: Optional[str], name: str) -> Path:
        path = FileIO.local_file(subdirectory, name, "png")
        return cls.save_bitmap_png(bitmap, path)

    @classmethod
    def save_local_bitmap_jpg(cls, bitmap: Any, subdirectory: Optional[str], name: str, quality: int = 95) -> Path:
        path = FileIO.local_file(subdirectory, name, "jpg")
        return cls.save_bitmap_jpg(bitmap, path, quality=quality)

    # ================================================================
    # OPENCV LOAD/SAVE (lazy import)
    # ================================================================

    @classmethod
    def load_opencv_image(cls, file_path: PathLike, flags: int = 1) -> Any:
        """
        Load an image via OpenCV as a numpy array.

        Common flags (cv2.IMREAD_*):
          - 1  IMREAD_COLOR: BGR (3 channels), drops alpha (default)
          - 0  IMREAD_GRAYSCALE: 1 channel grayscale
          - -1 IMREAD_UNCHANGED: keep original channels (e.g. BGRA if PNG has alpha)
        """
        import cv2  # lazy

        path = Path(file_path).resolve()
        if not path.is_file():
            base = path.with_suffix("")
            for ext in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".tif", ".tiff"]:
                attempt = base.with_suffix(ext)
                if attempt.is_file():
                    path = attempt
                    break

        if not path.is_file():
            raise FileNotFoundError(f"OpenCV image not found: {path}")

        img = cv2.imread(str(path), flags)
        if img is None:
            raise ValueError(f"OpenCV failed to read image: {path}")
        return img

    @classmethod
    def save_opencv_image_png(cls, image_bgr_or_gray: Any, file_path: PathLike) -> Path:
        import cv2  # lazy

        ok, buf = cv2.imencode(".png", image_bgr_or_gray)
        if not ok:
            raise IOError(f"OpenCV failed to encode PNG for: {file_path}")
        return FileIO.save(buf.tobytes(), file_path)

    @classmethod
    def save_opencv_image_jpg(cls, image_bgr_or_gray: Any, file_path: PathLike, quality: int = 95) -> Path:
        """
        Save an OpenCV image (numpy array) as JPG.

        quality: int in [0..100] (OpenCV expects IMWRITE_JPEG_QUALITY in this range)
        - We enforce the range strictly and raise if out of range.
        """
        if not isinstance(quality, int):
            raise TypeError("OpenCV JPG quality must be an int in [0..100].")
        if quality < 0 or quality > 100:
            raise ValueError(f"OpenCV JPG quality out of range: {quality} (expected 0..100).")

        import cv2  # lazy

        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, buf = cv2.imencode(".jpg", image_bgr_or_gray, params)
        if not ok:
            raise IOError(f"OpenCV failed to encode JPG for: {file_path}")
        return FileIO.save(buf.tobytes(), file_path)

    @classmethod
    def load_local_opencv_image(cls, subdirectory: Optional[str], name: str, extension: Optional[str] = None, flags: int = 1) -> Any:
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.load_opencv_image(path, flags=flags)

    @classmethod
    def save_local_opencv_image_png(cls, image_bgr_or_gray: Any, subdirectory: Optional[str], name: str) -> Path:
        path = FileIO.local_file(subdirectory, name, "png")
        return cls.save_opencv_image_png(image_bgr_or_gray, path)

    @classmethod
    def save_local_opencv_image_jpg(cls, image_bgr_or_gray: Any, subdirectory: Optional[str], name: str, quality: int = 95) -> Path:
        path = FileIO.local_file(subdirectory, name, "jpg")
        return cls.save_opencv_image_jpg(image_bgr_or_gray, path, quality=quality)

    # ================================================================
    # PATH STRING UTILS
    # ================================================================

    @classmethod
    def append_file_suffix(cls, file_name: str, suffix: str) -> str:
        p = Path(file_name)
        parent = p.parent
        stem = p.stem
        ext = p.suffix
        new_name = f"{stem}{suffix}{ext}"
        if str(parent) == ".":
            return new_name
        return str(parent / new_name)
