"""World-to-screen transform with zoom and pan."""
from __future__ import annotations


class Camera:
    """Maps world coordinates to screen pixels; the world centre is the default focus."""

    def __init__(self, world_size: tuple[float, float],
                 screen_size: tuple[int, int]) -> None:
        self.world_w, self.world_h = world_size
        self.screen_w, self.screen_h = screen_size
        self.zoom = 1.0
        self.focus = (self.world_w / 2.0, self.world_h / 2.0)

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        fx, fy = self.focus
        sx = (x - fx) * self.zoom + self.screen_w / 2.0
        sy = (y - fy) * self.zoom + self.screen_h / 2.0
        return int(sx), int(sy)

    def scale(self, length: float) -> int:
        return max(1, int(length * self.zoom))

    def zoom_by(self, factor: float) -> None:
        self.zoom = max(0.1, min(20.0, self.zoom * factor))

    def pan(self, dx: float, dy: float) -> None:
        fx, fy = self.focus
        self.focus = (min(max(fx + dx / self.zoom, 0.0), self.world_w),
                      min(max(fy + dy / self.zoom, 0.0), self.world_h))
