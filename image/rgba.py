# rgba.py

class RGBA:

    __slots__ = ("_r", "_g", "_b", "_a")

    def __init__(self, r: int, g: int, b: int, a: int = 255):
        self._r = self._clamp_int(r)
        self._g = self._clamp_int(g)
        self._b = self._clamp_int(b)
        self._a = self._clamp_int(a)

    # ------------------------------
    # Helpers
    # ------------------------------
    @staticmethod
    def _clamp_int(v):
        return max(0, min(255, int(v)))

    @staticmethod
    def _clamp_float(v):
        return max(0.0, min(1.0, float(v)))

    # ------------------------------
    # Integer accessors (0–255)
    # ------------------------------
    @property
    def ri(self):
        return self._r

    @ri.setter
    def ri(self, v):
        self._r = self._clamp_int(v)

    @property
    def gi(self):
        return self._g

    @gi.setter
    def gi(self, v):
        self._g = self._clamp_int(v)

    @property
    def bi(self):
        return self._b

    @bi.setter
    def bi(self, v):
        self._b = self._clamp_int(v)

    @property
    def ai(self):
        return self._a

    @ai.setter
    def ai(self, v):
        self._a = self._clamp_int(v)

    # ------------------------------
    # Float accessors (0.0–1.0)
    # ------------------------------
    @property
    def rf(self):
        return self._r / 255.0

    @rf.setter
    def rf(self, v):
        self._r = int(self._clamp_float(v) * 255)

    @property
    def gf(self):
        return self._g / 255.0

    @gf.setter
    def gf(self, v):
        self._g = int(self._clamp_float(v) * 255)

    @property
    def bf(self):
        return self._b / 255.0

    @bf.setter
    def bf(self, v):
        self._b = int(self._clamp_float(v) * 255)

    @property
    def af(self):
        return self._a / 255.0

    @af.setter
    def af(self, v):
        self._a = int(self._clamp_float(v) * 255)

   # ------------------------------
    # Utility
    # ------------------------------
    @classmethod
    def to_gray(cls, r, g, b) -> float:
        return (
            0.299 * r
            + 0.587 * g
            + 0.114 * b
        )
    
    def to_gray(self) -> int:
        gray = 0.299 * self._r + 0.587 * self._g + 0.114 * self._g
        return self._clamp_int(round(gray))
    
    def tuple(self):
        return (self._r, self._g, self._b, self._a)
    
    def __str__(self):
        return f"({self.rf:0.2f}, {self.gf:0.2f}, {self.bf:0.2f}, {self.af:0.2f})"

    def __repr__(self):
        return self.__str__()
    
    # ------------------------------
    # Blending helpers (OpenGL-style)
    # ------------------------------
    @staticmethod
    def blend_alpha(src: "RGBA", dst: "RGBA") -> "RGBA":
        """
        Classic alpha blending:
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        """
        sa = src.af
        da = dst.af

        out_a = sa + da * (1.0 - sa)
        if out_a == 0.0:
            return RGBA(0, 0, 0, 0)

        out_r = src.rf * sa + dst.rf * (1.0 - sa)
        out_g = src.gf * sa + dst.gf * (1.0 - sa)
        out_b = src.bf * sa + dst.bf * (1.0 - sa)

        return RGBA(
            int(out_r * 255),
            int(out_g * 255),
            int(out_b * 255),
            int(out_a * 255),
        )

    @staticmethod
    def blend_additive(src: "RGBA", dst: "RGBA") -> "RGBA":
        """
        Additive blending:
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        """
        sa = src.af
        da = dst.af

        out_a = min(1.0, sa + da)

        out_r = src.rf * sa + dst.rf
        out_g = src.gf * sa + dst.gf
        out_b = src.bf * sa + dst.bf

        # No manual clamp needed: RGBA(...) constructor clamps 0–255
        return RGBA(
            int(out_r * 255),
            int(out_g * 255),
            int(out_b * 255),
            int(out_a * 255),
        )