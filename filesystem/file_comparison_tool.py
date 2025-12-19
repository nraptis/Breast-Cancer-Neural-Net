from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Dict, List, Tuple, Union

from filesystem.file_io import FileIO, PathLike


class FileComparisonTool:
    """
    Compare files or images across two directories to detect duplicates.

    Supports:
      - raw file byte comparison (SHA-256)
      - image pixel comparison (same dims + same bytes)
      - local (project-root-relative) and absolute paths
      - PathLike (str | Path) everywhere
    """

    # ================================================================
    # Helpers
    # ================================================================

    @staticmethod
    def _to_path(p: PathLike) -> Path:
        return FileIO._to_path(p)

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def _collect_files(
        cls,
        directory: PathLike,
        extension: str | None,
        local: bool,
    ) -> List[Path]:
        if local:
            files = FileIO.get_all_files_local_recursive(directory)
        else:
            files = FileIO.get_all_files_recursive(directory)

        if extension:
            ext = extension.lower().lstrip(".")
            files = [p for p in files if p.suffix.lower() == f".{ext}"]

        return files

    # ================================================================
    # FILE BYTE COMPARISON
    # ================================================================

    @classmethod
    def _compare_file_lists(
        cls,
        files_a: List[Path],
        files_b: List[Path],
        label_a: str,
        label_b: str,
    ) -> None:
        print(f"[FileComparison] hashing {label_a}={len(files_a)} {label_b}={len(files_b)}")

        hashes_a: Dict[str, List[Path]] = {}
        for p in files_a:
            h = cls._hash_bytes(FileIO.load(p))
            hashes_a.setdefault(h, []).append(p)

        hashes_b: Dict[str, List[Path]] = {}
        for p in files_b:
            h = cls._hash_bytes(FileIO.load(p))
            hashes_b.setdefault(h, []).append(p)

        overlap = set(hashes_a.keys()) & set(hashes_b.keys())

        if not overlap:
            print("[FileComparison] ✅ no duplicate files detected")
            return

        print(f"[FileComparison] ❌ FOUND {len(overlap)} DUPLICATE FILE HASHES")

        for h in overlap:
            print("\nHASH:", h)
            print(f" {label_a}:")
            for p in hashes_a[h]:
                print("   ", p)
            print(f" {label_b}:")
            for p in hashes_b[h]:
                print("   ", p)

    @classmethod
    def compare_all_files(
        cls,
        directory_a: PathLike,
        directory_b: PathLike,
        extension: str | None = None,
    ) -> None:
        files_a = cls._collect_files(directory_a, extension, local=False)
        files_b = cls._collect_files(directory_b, extension, local=False)
        cls._compare_file_lists(files_a, files_b, "A", "B")

    @classmethod
    def compare_all_files_local(
        cls,
        directory_a: PathLike,
        directory_b: PathLike,
        extension: str | None = None,
    ) -> None:
        files_a = cls._collect_files(directory_a, extension, local=True)
        files_b = cls._collect_files(directory_b, extension, local=True)
        cls._compare_file_lists(files_a, files_b, "A(local)", "B(local)")

    # ================================================================
    # IMAGE PIXEL COMPARISON
    # ================================================================

    @classmethod
    def _compare_image_lists(
        cls,
        files_a: List[Path],
        files_b: List[Path],
        label_a: str,
        label_b: str,
    ) -> None:
        from PIL import Image  # lazy

        print(f"[ImageComparison] comparing {label_a}={len(files_a)} {label_b}={len(files_b)}")

        images_a: Dict[Tuple[int, int, str, bytes], List[Path]] = {}

        for p in files_a:
            try:
                im = Image.open(p)
                im.load()
                key = (im.width, im.height, im.mode, im.tobytes())
                images_a.setdefault(key, []).append(p)
            except Exception as e:
                print(f"[ImageComparison] ⚠️ failed to load {p}: {e}")

        matches = []

        for p in files_b:
            try:
                im = Image.open(p)
                im.load()
                key = (im.width, im.height, im.mode, im.tobytes())
                if key in images_a:
                    matches.append((images_a[key], p))
            except Exception as e:
                print(f"[ImageComparison] ⚠️ failed to load {p}: {e}")

        if not matches:
            print("[ImageComparison] ✅ no pixel-identical images detected")
            return

        print(f"[ImageComparison] ❌ FOUND {len(matches)} PIXEL-IDENTICAL IMAGES")

        for srcs, dup in matches:
            print("\nPIXEL MATCH:")
            print(f" {label_a}:")
            for s in srcs:
                print("   ", s)
            print(f" {label_b}:")
            print("   ", dup)

    @classmethod
    def compare_all_images(
        cls,
        directory_a: PathLike,
        directory_b: PathLike,
        extension: str | None = None,
    ) -> None:
        files_a = cls._collect_files(directory_a, extension, local=False)
        files_b = cls._collect_files(directory_b, extension, local=False)
        cls._compare_image_lists(files_a, files_b, "A", "B")

    @classmethod
    def compare_all_images_local(
        cls,
        directory_a: PathLike,
        directory_b: PathLike,
        extension: str | None = None,
    ) -> None:
        files_a = cls._collect_files(directory_a, extension, local=True)
        files_b = cls._collect_files(directory_b, extension, local=True)
        cls._compare_image_lists(files_a, files_b, "A(local)", "B(local)")
