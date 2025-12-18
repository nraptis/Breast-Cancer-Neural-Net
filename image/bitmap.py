# bitmap.py

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List, Tuple
import numpy as np
from PIL import Image
from image.rgba import RGBA
from typing import Optional
import torch
import torch.nn.functional as F
from image.pooling_mode import PoolingMode

from image.convolve_kernel_alignment import ConvolveKernelAlignment

from image.convolve_padding_mode import (
    ConvolvePaddingMode,
    ConvolvePaddingSame,
    ConvolvePaddingValid,
    ConvolvePaddingOffsetSame,
    ConvolvePaddingOffsetValid,
)

# ----------------------------------------------------------------------
# Bitmap: rgba[x][y] with OpenCV + Pillow interop
# ----------------------------------------------------------------------

class Bitmap:
    """
    Bitmap with:
        - width, height
        - pixels stored as rgba[x][y] where:
            x = 0..width-1  (left to right)
            y = 0..height-1 (top to bottom)
    """

    def __init__(self, width: int = 0, height: int = 0) -> None:
        self.width: int = 0
        self.height: int = 0
        self.rgba: List[List[RGBA]] = []  # rgba[x][y]
        if width > 0 and height > 0:
            self.allocate(width, height)

    # --------------------------------------------------
    # The ONLY place we allocate the internal rgba array
    # --------------------------------------------------
    def allocate(self, width: int, height: int) -> None:
        """
        Resize the bitmap and allocate internal storage.
        This is the ONLY place rgba[][] is allocated.
        """
        self.width = int(width)
        self.height = int(height)

        # rgba[x][y]
        self.rgba = [
            [RGBA(0, 0, 0, 255) for _y in range(self.height)]
            for _x in range(self.width)
        ]

    # --------------------------------------------------
    # Expansion / copy
    # --------------------------------------------------
    def expand(self, width: int, height: int) -> None:
        """
        Expand this bitmap to at least (width, height).

        Existing pixels are preserved in the top-left corner.
        Newly exposed pixels are filled with opaque black (0,0,0,255).

        If the requested size is smaller than or equal to the current
        size in both dimensions, this is a no-op (no shrinking).
        """
        new_w = int(width)
        new_h = int(height)

        if new_w <= self.width and new_h <= self.height:
            # Nothing to do; we only expand, never shrink.
            return

        old_w = self.width
        old_h = self.height
        old_rgba = self.rgba

        # Allocate new storage
        new_rgba = [
            [RGBA(0, 0, 0, 255) for _y in range(new_h)]
            for _x in range(new_w)
        ]

        copy_w = min(old_w, new_w)
        copy_h = min(old_h, new_h)

        # Copy old pixels into the new buffer (top-left aligned)
        for x in range(copy_w):
            src_col = old_rgba[x]
            dst_col = new_rgba[x]
            for y in range(copy_h):
                src_px = src_col[y]
                dst_px = dst_col[y]
                dst_px.ri = src_px.ri
                dst_px.gi = src_px.gi
                dst_px.bi = src_px.bi
                dst_px.ai = src_px.ai

        # Swap in the new buffer
        self.width = new_w
        self.height = new_h
        self.rgba = new_rgba

    def copy(self) -> "Bitmap":
        """
        Deep copy this bitmap into a new Bitmap instance.
        Pixels are duplicated (no shared RGBA objects).
        """
        result = Bitmap()
        result.allocate(self.width, self.height)
        new_rgba = result.rgba

        for x in range(self.width):
            src_col = self.rgba[x]
            dst_col = new_rgba[x]
            for y in range(self.height):
                src_px = src_col[y]
                dst_px = dst_col[y]
                dst_px.ri = src_px.ri
                dst_px.gi = src_px.gi
                dst_px.bi = src_px.bi
                dst_px.ai = src_px.ai

        return result

    # --------------------------------------------------
    # Loading Methods: load via FileIO + import_pillow
    # --------------------------------------------------
    
    @classmethod
    def with_image(cls, file_path) -> "Bitmap":
        """
        Convenience constructor: create a Bitmap and load an image from
        an explicit file path via FileIO.load_image.
        """
        bmp = cls()
        bmp.load_image(file_path)
        return bmp
    
    @classmethod
    def with_local_image(
        cls,
        subdirectory: str | None = None,
        name: str | None = None,
        extension: str | None = None,
    ) -> "Bitmap":
        """
        Convenience constructor: create a Bitmap and load an image using
        FileIO.load_local_image (which uses FileIO.local for path building).
        """
        bmp = cls()
        bmp.load_local_image(
            subdirectory=subdirectory,
            name=name,
            extension=extension,
        )
        return bmp

    def load_image(self, file_path):
        """
        Create a Bitmap from an explicit file path using FileIO.load_image.
        """
        from filesystem.file_utils import FileUtils
        image = FileUtils.load_image(file_path)
        self.import_pillow(image)
        return self  # optional, enables chaining

    def load_local_image(
        self,
        subdirectory: str | None = None,
        name: str | None = None,
        extension: str | None = None,
    ) -> "Bitmap":
        """
        Create a Bitmap using FileIO.load_local_image
        (which uses FileIO.local for path building).
        """
        from filesystem.file_utils import FileUtils
        image = FileUtils.load_local_image(
            subdirectory=subdirectory,
            name=name,
            extension=extension,
        )
        self.import_pillow(image)
        return self  # optional, enables chaining

    # --------------------------------------------------
    # Import from OpenCV (NumPy array)
    # --------------------------------------------------
    def import_opencv(self, mat: np.ndarray) -> None:
        """
        Import from an OpenCV-style NumPy array.
        Supports:
            - H x W (grayscale)
            - H x W x 3 (BGR)
            - H x W x 4 (BGRA)
        """
        if mat is None:
            raise ValueError("mat is None")

        if mat.ndim == 2:
            # Grayscale: shape = (H, W)
            h, w = mat.shape
            self.allocate(w, h)
            for y in range(h):
                for x in range(w):
                    v = int(mat[y, x])
                    self.rgba[x][y] = RGBA(v, v, v, 255)

        elif mat.ndim == 3:
            h, w, c = mat.shape
            if c not in (3, 4):
                raise ValueError(f"Unsupported channel count: {c}")

            self.allocate(w, h)

            if c == 3:
                # BGR
                for y in range(h):
                    for x in range(w):
                        b, g, r = mat[y, x]
                        self.rgba[x][y] = RGBA(int(r), int(g), int(b), 255)
            elif c == 4:
                # BGRA
                for y in range(h):
                    for x in range(w):
                        b, g, r, a = mat[y, x]
                        self.rgba[x][y] = RGBA(int(r), int(g), int(b), int(a))

        else:
            raise ValueError(f"Unsupported mat.ndim = {mat.ndim}")

    # --------------------------------------------------
    # Import from Pillow Image
    # --------------------------------------------------
    def import_pillow(self, image: Image.Image) -> None:
        """
        Import from a Pillow Image.
        Converts to RGBA first to simplify handling.
        """
        if image is None:
            raise ValueError("image is None")

        img = image.convert("RGBA")
        w, h = img.size
        self.allocate(w, h)

        pixels = img.load()
        for x in range(w):
            for y in range(h):
                r, g, b, a = pixels[x, y]
                self.rgba[x][y] = RGBA(int(r), int(g), int(b), int(a))

    # --------------------------------------------------
    # Export to OpenCV (NumPy array)
    # --------------------------------------------------
    def export_opencv(self) -> np.ndarray:
        """
        Export to an OpenCV-style NumPy array (H x W x 4, BGRA).
        Caller can convert to BGR if desired:
            bgr = bgra[:, :, :3]
        """
        h = self.height
        w = self.width
        mat = np.zeros((h, w, 4), dtype=np.uint8)

        for x in range(w):
            for y in range(h):
                px = self.rgba[x][y]
                # OpenCV expects B, G, R, A
                mat[y, x, 0] = px.bi
                mat[y, x, 1] = px.gi
                mat[y, x, 2] = px.ri
                mat[y, x, 3] = px.ai

        return mat

    # --------------------------------------------------
    # Export to Pillow Image
    # --------------------------------------------------
    def export_pillow(self) -> Image.Image:
        """
        Export to a Pillow RGBA Image.
        """
        img = Image.new("RGBA", (self.width, self.height))
        pixels = img.load()

        for x in range(self.width):
            for y in range(self.height):
                px = self.rgba[x][y]
                pixels[x, y] = (px.ri, px.gi, px.bi, px.ai)

        return img

    # --------------------------------------------------
    # Flood fill: set every pixel to the same RGBA color
    # --------------------------------------------------
    def flood(self, color: RGBA) -> None:
        """
        Set every pixel in this bitmap to the given RGBA color.

        If the bitmap has zero width or height, this is a no-op.
        """
        if self.width <= 0 or self.height <= 0:
            return

        # Use the int components from the input color.
        r = color.ri
        g = color.gi
        b = color.bi
        a = color.ai

        for x in range(self.width):
            col = self.rgba[x]
            for y in range(self.height):
                px = col[y]
                px.ri = r
                px.gi = g
                px.bi = b
                px.ai = a

    # --------------------------------------------------
    # Internal helper: compute overlap for stamping
    # --------------------------------------------------
    def _compute_stamp_bounds(self, glyph: "Bitmap", x: int, y: int):
        """
        Compute the overlapping region between this bitmap (destination)
        and the glyph bitmap (source), given that the glyph's top-left
        should be placed at (x, y) in destination coordinates.

        Returns:
            (start_dx, end_dx, start_dy, end_dy, start_gx, start_gy)
        or None if there is no overlap.
        """
        gw, gh = glyph.width, glyph.height
        dw, dh = self.width, self.height
        if gw <= 0 or gh <= 0 or dw <= 0 or dh <= 0:
            return None
        start_dx = max(x, 0)
        start_dy = max(y, 0)
        end_dx = min(x + gw, dw)
        end_dy = min(y + gh, dh)
        if start_dx >= end_dx or start_dy >= end_dy:
            return None
        start_gx = start_dx - x
        start_gy = start_dy - y
        return (start_dx, end_dx, start_dy, end_dy, start_gx, start_gy)

    # --------------------------------------------------
    # Stamp: overwrite pixels from glyph into this bitmap
    # --------------------------------------------------
    def stamp(self, glyph: "Bitmap", x: int, y: int) -> None:
        """
        Stamp `glyph` onto this bitmap so that glyph (0,0)
        lands at destination (x,y).

        For now, we simply REPLACE the destination pixels with
        the glyph pixels (no alpha blending).

        All edge/off-grid cases are handled gracefully:
        - If the stamp is fully off-screen, nothing happens.
        - If the stamp is partially off-screen, only the visible
            part is drawn.
        """
        bounds = self._compute_stamp_bounds(glyph, x, y)
        if bounds is None:
            return
        start_dx, end_dx, start_dy, end_dy, start_gx, start_gy = bounds
        for dy in range(start_dy, end_dy):
            gy = start_gy + (dy - start_dy)
            for dx in range(start_dx, end_dx):
                gx = start_gx + (dx - start_dx)
                self.rgba[dx][dy] = glyph.rgba[gx][gy]

    # --------------------------------------------------
    # Stamp with classic alpha
    # --------------------------------------------------
    def stamp_alpha(self, glyph: "Bitmap", x: int, y: int) -> None:
        bounds = self._compute_stamp_bounds(glyph, x, y)
        if bounds is None:
            return
        start_dx, end_dx, start_dy, end_dy, start_gx, start_gy = bounds
        for dy in range(start_dy, end_dy):
            gy = start_gy + (dy - start_dy)
            for dx in range(start_dx, end_dx):
                gx = start_gx + (dx - start_dx)
                src_px = glyph.rgba[gx][gy]
                dst_px = self.rgba[dx][dy]
                self.rgba[dx][dy] = RGBA.blend_alpha(src_px, dst_px)
                
    # --------------------------------------------------
    # Stamp with additive blending
    # --------------------------------------------------
    def stamp_additive(self, glyph: "Bitmap", x: int, y: int) -> None:
        bounds = self._compute_stamp_bounds(glyph, x, y)
        if bounds is None:
            return
        start_dx, end_dx, start_dy, end_dy, start_gx, start_gy = bounds
        for dy in range(start_dy, end_dy):
            gy = start_gy + (dy - start_dy)
            for dx in range(start_dx, end_dx):
                gx = start_gx + (dx - start_dx)
                src_px = glyph.rgba[gx][gy]
                dst_px = self.rgba[dx][dy]
                self.rgba[dx][dy] = RGBA.blend_additive(src_px, dst_px)

    # --------------------------------------------------
    # Crop a sub-rectangle into a new Bitmap
    # --------------------------------------------------
    def crop(
        self,
        x: int,
        y: int,
        width: int,
        height: int) -> "Bitmap":
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0 or self.width <= 0 or self.height <= 0:
            return Bitmap()
        gw, gh = self.width, self.height
        dw, dh = width, height
        x_offset = -x
        y_offset = -y
        start_dx = max(x_offset, 0)
        start_dy = max(y_offset, 0)
        end_dx = min(x_offset + gw, dw)
        end_dy = min(y_offset + gh, dh)
        if start_dx >= end_dx or start_dy >= end_dy:
            return Bitmap()
        start_gx = start_dx - x_offset
        start_gy = start_dy - y_offset
        crop_w = end_dx - start_dx
        crop_h = end_dy - start_dy
        result = Bitmap(crop_w, crop_h)
        for dy in range(crop_h):
            sy = start_gy + dy
            for dx in range(crop_w):
                sx = start_gx + dx
                result.rgba[dx][dy] = self.rgba[sx][sy]
        return result
    
    # --------------------------------------------------
    # Helper: integer ceil-div (no floats)
    # --------------------------------------------------
    @staticmethod
    def _ceil_div(n: int, d: int) -> int:
        n = int(n)
        d = int(d)
        if d <= 0:
            return 0
        return n // d + (1 if (n % d) != 0 else 0)
    
    # --------------------------------------------------
    # Convolution frame helper (NON-THROWING, Torch-ish + your OFFSET contract)
    # --------------------------------------------------
    @classmethod
    def convolve_frame(
        cls,
        image_width: int,
        image_height: int,
        kernel_width: int,
        kernel_height: int,
        offset_x: int = 0,
        offset_y: int = 0,
        stride_h: int = 1,
        stride_v: int = 1,
        dilation_h: int = 1,
        dilation_v: int = 1,
        kernel_alignment: ConvolveKernelAlignment = ConvolveKernelAlignment.CENTER,
        padding_mode: ConvolvePaddingMode = ConvolvePaddingValid(),
    ) -> Tuple[int, int, int, int]:
        W = int(image_width)
        H = int(image_height)
        kw = int(kernel_width)
        kh = int(kernel_height)

        ox = int(offset_x)
        oy = int(offset_y)
        sh = int(stride_h)
        sv = int(stride_v)
        dh = int(dilation_h)
        dv = int(dilation_v)

        if W <= 0 or H <= 0:
            return (0, 0, 0, 0)
        if kw <= 0 or kh <= 0:
            return (0, 0, 0, 0)
        if sh <= 0 or sv <= 0:
            return (0, 0, 0, 0)
        if dh <= 0 or dv <= 0:
            return (0, 0, 0, 0)

        # Effective kernel footprint under dilation
        k_eff_w = (kw - 1) * dh + 1
        k_eff_h = (kh - 1) * dv + 1

        # --------------------------------------------------
        # OFFSET contract enforcement + base shift
        # --------------------------------------------------
        base_x = 0
        base_y = 0
        max_ox = 0
        max_oy = 0

        is_offset_mode = isinstance(padding_mode, (ConvolvePaddingOffsetSame, ConvolvePaddingOffsetValid))
        if (ox != 0 or oy != 0) and not is_offset_mode:
            return (0, 0, 0, 0)

        if is_offset_mode:
            max_ox = int(padding_mode.max_offset_x)
            max_oy = int(padding_mode.max_offset_y)
            if max_ox < 0 or max_oy < 0:
                return (0, 0, 0, 0)
            if abs(ox) > max_ox or abs(oy) > max_oy:
                return (0, 0, 0, 0)

            base_x = max_ox
            base_y = max_oy

        # Budgeted footprint (offset expands footprint)
        k_budget_w = k_eff_w + 2 * max_ox
        k_budget_h = k_eff_h + 2 * max_oy

        # --------------------------------------------------
        # Output + padding (Torch-ish sizing)
        # --------------------------------------------------
        pad_left = pad_right = pad_top = pad_bottom = 0

        is_same = isinstance(padding_mode, (ConvolvePaddingSame, ConvolvePaddingOffsetSame))
        if is_same:
            out_w = cls._ceil_div(W, sh)
            out_h = cls._ceil_div(H, sv)

            pad_total_w = max(0, (out_w - 1) * sh + k_budget_w - W)
            pad_total_h = max(0, (out_h - 1) * sv + k_budget_h - H)

            pad_left = pad_total_w // 2
            pad_right = pad_total_w - pad_left
            pad_top = pad_total_h // 2
            pad_bottom = pad_total_h - pad_top
        else:
            # VALID (including offset-valid)
            if W < k_budget_w or H < k_budget_h:
                return (0, 0, 0, 0)

            out_w = (W - k_budget_w) // sh + 1
            out_h = (H - k_budget_h) // sv + 1

        if out_w <= 0 or out_h <= 0:
            return (0, 0, 0, 0)

        padded_w = W + pad_left + pad_right
        padded_h = H + pad_top + pad_bottom

        # --------------------------------------------------
        # IMPORTANT: sampling start is ALWAYS TOP_LEFT.
        # This matches Bitmap.convolve() exactly.
        # --------------------------------------------------
        start_x0 = base_x + ox
        start_y0 = base_y + oy

        # Bounds check for the TOP_LEFT sampler
        min_x = start_x0
        min_y = start_y0
        max_x = start_x0 + (out_w - 1) * sh + (kw - 1) * dh
        max_y = start_y0 + (out_h - 1) * sv + (kh - 1) * dv

        if min_x < 0 or min_y < 0 or max_x >= padded_w or max_y >= padded_h:
            return (0, 0, 0, 0)

        # --------------------------------------------------
        # Now: report the chosen anchor position in SOURCE coords.
        # (this does NOT affect validity or sampling)
        # In padded space, anchor = start + (ax,ay)*dilation
        # In source space, subtract SAME padding shift.
        # --------------------------------------------------
        ax, ay = ConvolveKernelAlignment.anchor(kernel_alignment, kw, kh)

        anchor_x0 = start_x0 + ax * dh
        anchor_y0 = start_y0 + ay * dv

        anchor_x_src = anchor_x0 - pad_left
        anchor_y_src = anchor_y0 - pad_top

        return (int(anchor_x_src), int(anchor_y_src), int(out_w), int(out_h))
    
    @classmethod
    def convolve_frame_mask(
        cls,
        image_width: int,
        image_height: int,
        mask: List[List[float]],
        offset_x: int = 0,
        offset_y: int = 0,
        stride_h: int = 1,
        stride_v: int = 1,
        dilation_h: int = 1,
        dilation_v: int = 1,
        kernel_alignment: ConvolveKernelAlignment = ConvolveKernelAlignment.CENTER,
        padding_mode: ConvolvePaddingMode = ConvolvePaddingValid(),
    ) -> Tuple[int, int, int, int]:
        if mask is None:
            return (0, 0, 0, 0)

        kw = len(mask)
        if kw <= 0:
            return (0, 0, 0, 0)

        col0 = mask[0]
        kh = len(col0) if col0 is not None else 0
        if kh <= 0:
            return (0, 0, 0, 0)

        return cls.convolve_frame(
            image_width=image_width,
            image_height=image_height,
            kernel_width=kw,
            kernel_height=kh,
            offset_x=offset_x,
            offset_y=offset_y,
            stride_h=stride_h,
            stride_v=stride_v,
            dilation_h=dilation_h,
            dilation_v=dilation_v,
            kernel_alignment=kernel_alignment,
            padding_mode=padding_mode,
        )
    
    def convolve(
        self,
        mask: List[List[float]],
        offset_x: int = 0,
        offset_y: int = 0,
        stride_h: int = 1,
        stride_v: int = 1,
        dilation_h: int = 1,
        dilation_v: int = 1,
        padding_mode: ConvolvePaddingMode = ConvolvePaddingValid(),
    ) -> "Bitmap":
        # Frame is NON-THROWING. We ignore (x,y) and always request TOP_LEFT.
        _, _, out_w, out_h = Bitmap.convolve_frame_mask(
            self.width,
            self.height,
            mask,
            offset_x=offset_x,
            offset_y=offset_y,
            stride_h=stride_h,
            stride_v=stride_v,
            dilation_h=dilation_h,
            dilation_v=dilation_v,
            kernel_alignment=ConvolveKernelAlignment.TOP_LEFT,
            padding_mode=padding_mode,
        )
        if out_w <= 0 or out_h <= 0:
            return Bitmap()

        W = int(self.width)
        H = int(self.height)

        kw = len(mask)
        kh = len(mask[0])

        ox = int(offset_x)
        oy = int(offset_y)
        sh = int(stride_h)
        sv = int(stride_v)
        dh = int(dilation_h)
        dv = int(dilation_v)

        k_eff_w = (kw - 1) * dh + 1
        k_eff_h = (kh - 1) * dv + 1

        # OFFSET base shift (same as before)
        base_x = 0
        base_y = 0
        max_ox = 0
        max_oy = 0
        is_offset_mode = isinstance(padding_mode, (ConvolvePaddingOffsetSame, ConvolvePaddingOffsetValid))
        if is_offset_mode:
            max_ox = int(padding_mode.max_offset_x)
            max_oy = int(padding_mode.max_offset_y)
            base_x = max_ox
            base_y = max_oy

        k_budget_w = k_eff_w + 2 * max_ox
        k_budget_h = k_eff_h + 2 * max_oy

        # SAME padding build (unchanged)
        is_same = isinstance(padding_mode, (ConvolvePaddingSame, ConvolvePaddingOffsetSame))
        if is_same:
            out_w_same = Bitmap._ceil_div(W, sh)
            out_h_same = Bitmap._ceil_div(H, sv)

            pad_total_w = max(0, (out_w_same - 1) * sh + k_budget_w - W)
            pad_total_h = max(0, (out_h_same - 1) * sv + k_budget_h - H)

            pad_left = pad_total_w // 2
            pad_right = pad_total_w - pad_left
            pad_top = pad_total_h // 2
            pad_bottom = pad_total_h - pad_top

            padded = Bitmap(W + pad_left + pad_right, H + pad_top + pad_bottom)
            padded.flood(RGBA(0, 0, 0, 0))
            padded.stamp(self, pad_left, pad_top)

            src_f = padded.export_opencv().astype(np.float32, copy=False)

            start_x0 = base_x + ox
            start_y0 = base_y + oy
        else:
            src_f = self.export_opencv().astype(np.float32, copy=False)

            start_x0 = base_x + ox
            start_y0 = base_y + oy

        # mask[x][y] -> mask_np[ky,kx]
        mask_np = np.empty((kh, kw), dtype=np.float32)
        for x in range(kw):
            col = mask[x]
            for y in range(kh):
                mask_np[y, x] = float(col[y])

        acc = np.zeros((out_h, out_w, 4), dtype=np.float32)

        # TOP_LEFT sampling only (unchanged)
        for ky in range(kh):
            sy0 = start_y0 + ky * dv
            sy1 = sy0 + out_h * sv
            for kx in range(kw):
                w = float(mask_np[ky, kx])
                if w == 0.0:
                    continue
                sx0 = start_x0 + kx * dh
                sx1 = sx0 + out_w * sh
                acc += src_f[sy0:sy1:sv, sx0:sx1:sh, :] * w

        out_u8 = np.clip(acc + 0.5, 0.0, 255.0).astype(np.uint8)
        result = Bitmap(out_w, out_h)
        result.import_opencv(out_u8)
        return result



    # --------------------------------------------------
    # Pooling frame helper (STRICT)
    # --------------------------------------------------
    @classmethod
    def pool_frame(
        cls,
        image_width: int,
        image_height: int,
        kernel_width: int = 2,
        kernel_height: int = 2,
        stride_h: int = 0,
        stride_v: int = 0,
    ) -> Tuple[int, int, int, int]:
        W = int(image_width)
        H = int(image_height)

        kw = int(kernel_width)
        kh = int(kernel_height)
        sh = int(stride_h) if int(stride_h) > 0 else kw
        sv = int(stride_v) if int(stride_v) > 0 else kh

        if W <= 0 or H <= 0:
            raise ValueError(f"pool_frame: invalid image size ({W},{H})")
        if kw <= 0 or kh <= 0:
            raise ValueError(f"pool_frame: kernel must be > 0 (kw={kw}, kh={kh})")
        if sh <= 0 or sv <= 0:
            raise ValueError(f"pool_frame: stride must be > 0 (sh={sh}, sv={sv})")

        out_w = (W - kw) // sh + 1
        out_h = (H - kh) // sv + 1

        if out_w <= 0 or out_h <= 0:
            raise ValueError(
                f"pool_frame: non-positive output ({out_w},{out_h}) "
                f"from image=({W},{H}) kernel=({kw},{kh}) stride=({sh},{sv})"
            )

        return (0, 0, int(out_w), int(out_h))
        
    def pool(
        self,
        kernel_width: int = 2,
        kernel_height: int = 2,
        stride_h: int = 0,
        stride_v: int = 0,
        mode: PoolingMode = PoolingMode.MAX_PER_CHANNEL,
    ) -> "Bitmap":
        kw = int(kernel_width)
        kh = int(kernel_height)
        sh = int(stride_h) if int(stride_h) > 0 else kw
        sv = int(stride_v) if int(stride_v) > 0 else kh

        # STRICT: if the frame can't be computed, pool_frame raises and we let it propagate.
        _, _, out_w, out_h = Bitmap.pool_frame(self.width, self.height, kw, kh, sh, sv)

        src_u8 = self.export_opencv()  # H x W x 4 (BGRA)
        src_f32 = src_u8.astype(np.float32, copy=False)

        out = np.zeros((out_h, out_w, 4), dtype=np.float32)

        for oy in range(out_h):
            y0 = oy * sv
            y1 = y0 + kh
            for ox in range(out_w):
                x0 = ox * sh
                x1 = x0 + kw

                window_f32 = src_f32[y0:y1, x0:x1, :]  # kh x kw x 4
                window_u8  = src_u8[y0:y1, x0:x1, :]   # kh x kw x 4

                if mode == PoolingMode.MIN_PER_CHANNEL:
                    out[oy, ox, :] = window_f32.reshape(-1, 4).min(axis=0)

                elif mode == PoolingMode.AVERAGE_PER_CHANNEL:
                    out[oy, ox, :] = window_f32.reshape(-1, 4).mean(axis=0)

                elif mode == PoolingMode.MAX_PIXEL_BY_RGB_SUM:
                    flat = window_u8.reshape(-1, 4)  # BGRA (uint8)
                    rgb_sum = (
                        flat[:, 0].astype(np.int32) +
                        flat[:, 1].astype(np.int32) +
                        flat[:, 2].astype(np.int32)
                    )
                    idx = int(np.argmax(rgb_sum))
                    out[oy, ox, :] = flat[idx].astype(np.float32)

                elif mode == PoolingMode.MIN_PIXEL_BY_RGB_SUM:
                    flat = window_u8.reshape(-1, 4)
                    rgb_sum = (
                        flat[:, 0].astype(np.int32) +
                        flat[:, 1].astype(np.int32) +
                        flat[:, 2].astype(np.int32)
                    )
                    idx = int(np.argmin(rgb_sum))
                    out[oy, ox, :] = flat[idx].astype(np.float32)

                else:
                    # Default: MAX_PER_CHANNEL (also handles unknown modes)
                    out[oy, ox, :] = window_f32.reshape(-1, 4).max(axis=0)

        out_u8 = np.clip(out + 0.5, 0.0, 255.0).astype(np.uint8)

        result = Bitmap(out_w, out_h)
        result.import_opencv(out_u8)
        return result
    
    def pool_torch(
        self,
        kernel_width: int = 2,
        kernel_height: int = 2,
        stride_h: int = 0,
        stride_v: int = 0,
        mode: PoolingMode = PoolingMode.MAX_PER_CHANNEL,
        device: str = "cpu",
    ) -> "Bitmap":
        kw = int(kernel_width)
        kh = int(kernel_height)
        sh = int(stride_h) if int(stride_h) > 0 else kw
        sv = int(stride_v) if int(stride_v) > 0 else kh

        # STRICT: will raise if invalid / non-positive output
        _, _, out_w, out_h = Bitmap.pool_frame(self.width, self.height, kw, kh, sh, sv)

        # Export BGRA u8
        src_bgra_u8 = self.export_opencv()  # (H,W,4) uint8 BGRA
        H, W = src_bgra_u8.shape[0], src_bgra_u8.shape[1]

        # Torch: [1,4,H,W]
        x_u8 = torch.from_numpy(src_bgra_u8).to(device=device)
        x_u8 = x_u8.permute(2, 0, 1).unsqueeze(0).contiguous()  # uint8

        # Build windows: [1,4,out_h,out_w,kh,kw]
        # (unfold dims are exact VALID semantics given the out_w/out_h we computed)
        win_u8 = x_u8.unfold(2, kh, sv).unfold(3, kw, sh)

        # Helper: flatten window pixels to last dim = kh*kw
        # flat_u8: [1,4,out_h,out_w,kh*kw]
        flat_u8 = win_u8.contiguous().view(1, 4, out_h, out_w, kh * kw)

        if mode == PoolingMode.MIN_PER_CHANNEL:
            out_f = flat_u8.float().amin(dim=-1)  # [1,4,out_h,out_w]

        elif mode == PoolingMode.AVERAGE_PER_CHANNEL:
            # float32 mean like NumPy window_f32.mean(axis=0)
            out_f = flat_u8.float().mean(dim=-1)  # [1,4,out_h,out_w]

        elif mode == PoolingMode.MAX_PIXEL_BY_RGB_SUM or mode == PoolingMode.MIN_PIXEL_BY_RGB_SUM:
            # RGB sum uses BGRA channels 0,1,2
            # rgb_sum: [1,out_h,out_w,kh*kw] int32
            rgb_sum = (
                flat_u8[:, 0, :, :, :].to(torch.int32) +
                flat_u8[:, 1, :, :, :].to(torch.int32) +
                flat_u8[:, 2, :, :, :].to(torch.int32)
            )

            if mode == PoolingMode.MAX_PIXEL_BY_RGB_SUM:
                idx = torch.argmax(rgb_sum, dim=-1)  # [1,out_h,out_w]
            else:
                idx = torch.argmin(rgb_sum, dim=-1)  # [1,out_h,out_w]

            # Gather BGRA at that pixel index for each channel
            # idx_exp: [1,1,out_h,out_w,1] then broadcast along channel dim by repeating gather per-channel
            idx_exp = idx.unsqueeze(1).unsqueeze(-1)  # [1,1,out_h,out_w,1]

            gathered = torch.gather(flat_u8, dim=-1, index=idx_exp.expand(1, 4, out_h, out_w, 1))
            out_f = gathered.squeeze(-1).float()  # [1,4,out_h,out_w] float

        else:
            # Default MAX_PER_CHANNEL (also covers unknown modes, like your NumPy code)
            out_f = flat_u8.float().amax(dim=-1)  # [1,4,out_h,out_w]

        # Match NumPy rounding+clamp: clip(out + 0.5) then uint8
        out_u8 = torch.clamp(out_f + 0.5, 0.0, 255.0).to(torch.uint8)

        # Back to numpy BGRA (H,W,4)
        out_bgra_u8 = (
            out_u8.squeeze(0)
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )

        result = Bitmap(out_w, out_h)
        result.import_opencv(out_bgra_u8)
        return result

    def compare(self, bitmap: Optional["Bitmap"], tolerance: int = 0) -> bool:
        """
        Compare this bitmap against another bitmap.

        Returns True if:
        - bitmap is not None
        - dimensions match
        - all RGBA channel differences are <= tolerance

        tolerance is an absolute per-channel tolerance (0 = exact match).
        """
        if bitmap is None:
            return False

        if self.width != bitmap.width or self.height != bitmap.height:
            return False

        for x in range(self.width):
            col_a = self.rgba[x]
            col_b = bitmap.rgba[x]
            for y in range(self.height):
                a = col_a[y]
                b = col_b[y]

                if abs(a.ri - b.ri) > tolerance:
                    return False
                if abs(a.gi - b.gi) > tolerance:
                    return False
                if abs(a.bi - b.bi) > tolerance:
                    return False
                if abs(a.ai - b.ai) > tolerance:
                    return False

        return True
    

    def padding_mode_string(padding_mode) -> str:
        if isinstance(padding_mode, ConvolvePaddingSame):
            return "SAME"
        if isinstance(padding_mode, ConvolvePaddingValid):
            return "VALID"
        if isinstance(padding_mode, ConvolvePaddingOffsetSame):
            return f"OFFSET_SAME(max_offset_x={padding_mode.max_offset_x}, max_offset_y={padding_mode.max_offset_y})"
        if isinstance(padding_mode, ConvolvePaddingOffsetValid):
            return f"OFFSET_VALID(max_offset_x={padding_mode.max_offset_x}, max_offset_y={padding_mode.max_offset_y})"
        return padding_mode.__class__.__name__