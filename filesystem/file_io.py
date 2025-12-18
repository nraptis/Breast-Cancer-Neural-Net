# file_io.py

from __future__ import annotations

from pathlib import Path
from typing import Union, List, Tuple

PathLike = Union[str, Path]
FileSpec = Tuple[List[str], str]

class FileIO:

    # ================================================================
    # Path builders: directories + files
    # ================================================================

    @classmethod
    def local_directory(cls, subdirectory: PathLike | None = None) -> Path:
        """
        Build a fully-qualified directory path inside the project root.
        Strips any leading/trailing slashes.
        """
        project_root = Path(__file__).resolve().parent.parent

        if subdirectory:
            subdirectory = cls._strip_outer_slashes(str(subdirectory))
            return (project_root / subdirectory).resolve()

        return project_root

    @classmethod
    def local_file(
        cls,
        subdirectory: str | None = None,
        name: str | None = None,
        extension: str | None = None,
    ) -> Path:
        """
        Build a fully-resolved file path inside the project using the
        familiar subdirectory + name + extension pattern.

        Example:
            local_file("images", "cat", "png")
            -> ROOT/images/cat.png
        """
        if name is None or len(name.strip()) == 0:
            raise ValueError("local_file requires a non-empty 'name'")

        # Clean outer slashes
        if subdirectory:
            subdirectory = cls._strip_outer_slashes(subdirectory)
        name = cls._strip_outer_slashes(name)

        name_path = Path(name)

        # Extension override
        if extension:
            extension = extension.lstrip(".")
            parent = name_path.parent
            stem = name_path.stem

            if str(parent) == ".":
                file_name = f"{stem}.{extension}"
            else:
                file_name = str(parent / f"{stem}.{extension}")
        else:
            file_name = name

        root = Path(__file__).resolve().parent.parent
        if subdirectory:
            final_path = root / subdirectory / file_name
        else:
            final_path = root / file_name

        return final_path.resolve()

    # ================================================================
    # FILE LISTING (FLAT + RECURSIVE)
    # ================================================================

    @classmethod
    def get_all_files(cls, directory: PathLike) -> List[Path]:
        """
        Return all direct (non-directory) files in directory.
        """
        dir_path = cls._to_path(directory)
        if not dir_path.exists():
            return []
        return [p for p in dir_path.iterdir() if p.is_file()]

    @classmethod
    def get_all_files_local(cls, subdirectory: PathLike) -> List[Path]:
        folder = cls.local_directory(subdirectory)
        return cls.get_all_files(folder)

    @classmethod
    def get_all_files_recursive(cls, directory: PathLike) -> List[Path]:
        """
        Return all files in all nested subdirectories.
        """
        dir_path = cls._to_path(directory)
        if not dir_path.exists():
            return []
        return [p for p in dir_path.rglob("*") if p.is_file()]

    @classmethod
    def get_all_files_local_recursive(cls, subdirectory: PathLike) -> List[Path]:
        folder = cls.local_directory(subdirectory)
        return cls.get_all_files_recursive(folder)
    
    @classmethod
    def get_all_files_specs_local_recursive(cls, subdirectory: PathLike) -> tuple[List[str], str]:

        # uses get_all_files_local_recursive

        # strip this out
        # project_root = Path(__file__).resolve().parent.parent

        #splits on /

        #return in format tuple[List[str], str]:


        folder = cls.local_directory(subdirectory)
        return cls.get_all_files_recursive(folder)

    # ================================================================
    # Helpers
    # ================================================================

    @classmethod
    def _strip_outer_slashes(cls, s: str) -> str:
        if not isinstance(s, str):
            raise TypeError("Path must be a string")
        return s.lstrip("/\\").rstrip("/\\")

    @classmethod
    def _ensure_parent_dir(cls, path: Path) -> None:
        parent = path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _to_path(cls, p: PathLike) -> Path:
        if isinstance(p, Path):
            return p.resolve()
        return Path(p).resolve()

    # ================================================================
    # Core I/O: bytes only
    # ================================================================

    @classmethod
    def load(cls, file_path: PathLike) -> bytes:
        """
        Load raw bytes from an explicit path.

        Caller decides how to interpret the bytes (text, image, etc.).
        """
        path = cls._to_path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        return path.read_bytes()

    @classmethod
    def load_local(
        cls,
        subdirectory: str | None = None,
        name: str | None = None,
        extension: str | None = None,
    ) -> bytes:
        """
        Build a local file path via local_file() and load raw bytes.
        """
        path = cls.local_file(subdirectory=subdirectory, name=name, extension=extension)
        return cls.load(path)

    @classmethod
    def save(cls, data: bytes, file_path: PathLike) -> Path:
        """
        Save raw bytes to a specific path.

        - Creates parent directories if needed.
        - Returns the resolved Path actually written.
        """
        path = cls._to_path(file_path)
        cls._ensure_parent_dir(path)
        path.write_bytes(data)
        return path

    @classmethod
    def save_local(
        cls,
        data: bytes,
        subdirectory: str | None = None,
        name: str | None = None,
        extension: str | None = None,
    ) -> Path:
        """
        Build a local file path via local_file() and save raw bytes there.
        """
        path = cls.local_file(subdirectory=subdirectory, name=name, extension=extension)
        return cls.save(data, path)
