from __future__ import annotations

from pathlib import Path
from typing import List, Union

from sklearn.model_selection import train_test_split

from filesystem.file_io import FileIO
from filesystem.file_utils import FileUtils

PathLike = Union[str, Path]


class Splitter:
    """
    Folder-based train/validate/test splitter.

    Fast path:
      - load with Pillow
      - save as PNG
      - no mkdir anywhere (FileIO.save creates parent dirs)

    Output:
      data_split/
        train/{benign,malignant}
        validate/{benign,malignant}
        test/{benign,malignant}
    """

    OUT_ROOT = "data_split"

    TEST_FRAC = 0.10
    VAL_FRAC = 0.10
    SEED = 911911

    PROGRESS_EVERY = 100

    @classmethod
    def go(cls, benign: List[PathLike], malignant: List[PathLike]) -> None:
        benign_paths = sorted([Path(p) for p in benign])
        malignant_paths = sorted([Path(p) for p in malignant])

        cls._verify_inputs(benign_paths, malignant_paths)

        benign_trainval, benign_test = train_test_split(
            benign_paths, test_size=cls.TEST_FRAC, random_state=cls.SEED, shuffle=True
        )
        malignant_trainval, malignant_test = train_test_split(
            malignant_paths, test_size=cls.TEST_FRAC, random_state=cls.SEED, shuffle=True
        )

        val_frac_rel = cls.VAL_FRAC / (1.0 - cls.TEST_FRAC)

        benign_train, benign_val = train_test_split(
            benign_trainval, test_size=val_frac_rel, random_state=cls.SEED + 1, shuffle=True
        )
        malignant_train, malignant_val = train_test_split(
            malignant_trainval, test_size=val_frac_rel, random_state=cls.SEED + 2, shuffle=True
        )

        cls._write_split("train", "benign", benign_train)
        cls._write_split("train", "malignant", malignant_train)

        cls._write_split("validate", "benign", benign_val)
        cls._write_split("validate", "malignant", malignant_val)

        cls._write_split("test", "benign", benign_test)
        cls._write_split("test", "malignant", malignant_test)

        cls._print_counts(
            benign_train, benign_val, benign_test,
            malignant_train, malignant_val, malignant_test
        )

    @classmethod
    def _verify_inputs(cls, benign: List[Path], malignant: List[Path]) -> None:
        if len(benign) == 0 or len(malignant) == 0:
            raise ValueError("Empty benign or malignant list.")

        # Prevent overwrites within each class (flat output dirs)
        b_names = [p.name for p in benign]
        m_names = [p.name for p in malignant]
        if len(set(b_names)) != len(b_names):
            raise ValueError("Duplicate filenames found within benign list.")
        if len(set(m_names)) != len(m_names):
            raise ValueError("Duplicate filenames found within malignant list.")

    @classmethod
    def _write_split(cls, split: str, label: str, files: List[Path]) -> None:
        """
        Clears existing files in the split folder (if it exists),
        then writes PNGs using Pillow via FileUtils.
        """
        out_dir = FileIO.local_directory(f"{cls.OUT_ROOT}/{split}/{label}")

        # Clear existing files so reruns are clean (no mkdir)
        if out_dir.exists():
            for existing in out_dir.glob("*"):
                if existing.is_file() or existing.is_symlink():
                    existing.unlink()

        subdir = f"{cls.OUT_ROOT}/{split}/{label}"
        total = len(files)

        for i, src in enumerate(files):
            last_name = cls._write_one_png_pillow(src, subdir=subdir)

            # progress
            done = i + 1
            if done % cls.PROGRESS_EVERY == 0 or done == total:
                print(f"[Splitter] {split}/{label}: {done}/{total} last={last_name}")

    @classmethod
    def _write_one_png_pillow(cls, src: Path, subdir: str) -> str:
        """
        Load with Pillow, save as PNG.
        Returns the last written file name for progress printing.
        """
        im = FileUtils.load_pillow_image(src)
        FileUtils.save_local_pillow_image_png(im, subdirectory=subdir, name=src.stem)
        return f"{src.stem}.png"

    @classmethod
    def _print_counts(
        cls,
        b_tr: List[Path], b_va: List[Path], b_te: List[Path],
        m_tr: List[Path], m_va: List[Path], m_te: List[Path],
    ) -> None:
        print("Split counts:")
        print(f"  train:    benign={len(b_tr)} malignant={len(m_tr)} total={len(b_tr)+len(m_tr)}")
        print(f"  validate: benign={len(b_va)} malignant={len(m_va)} total={len(b_va)+len(m_va)}")
        print(f"  test:     benign={len(b_te)} malignant={len(m_te)} total={len(b_te)+len(m_te)}")
