"""Outdoor complex terrains for Unitree Go2.

Includes:
- Marches (Pyramid stairs, inverted stairs, random stairs)
- Terrains accidentés (Perlin noise outdoor hills, random rough, wave terrain, tilted grid)
- Murs (Wall obstacles, barriers, slalom/maze walls)
- Longs couloirs (Long narrow corridors, parallel lanes)
- Trous (Pits, trenches, stepping stones over void, narrow beams over void)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import mujoco
import numpy as np

from mjlab.terrains.primitive_terrains import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
  BoxFlatTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  BoxInvertedPyramidStairsTerrainCfg,
  BoxRandomStairsTerrainCfg,
  BoxSteppingStonesTerrainCfg,
  BoxNarrowBeamsTerrainCfg,
  BoxTiltedGridTerrainCfg,
  make_border,
  brand_ramp,
  darken_rgba,
  _MUJOCO_BLUE,
  _MUJOCO_GREEN,
)
from mjlab.terrains.heightfield_terrains import (
  HfPerlinNoiseTerrainCfg,
  HfRandomUniformTerrainCfg,
  HfWaveTerrainCfg,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg


@dataclass(kw_only=True)
class BoxCorridorTerrainCfg(SubTerrainCfg):
  """Long corridors bounded by walls (Longs couloirs).
  
  Forms long narrow corridors along the X direction with openings,
  forcing the robot to navigate confined paths without colliding with walls.
  """
  corridor_width_range: tuple[float, float] = (1.6, 0.9)
  wall_height_range: tuple[float, float] = (0.7, 1.2)
  wall_thickness: float = 0.15
  border_width: float = 0.25
  platform_width: float = 1.4

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []

    wall_h = self.wall_height_range[0] + difficulty * (
      self.wall_height_range[1] - self.wall_height_range[0]
    )
    corr_w = self.corridor_width_range[0] - difficulty * (
      self.corridor_width_range[0] - self.corridor_width_range[1]
    )

    floor_h = 0.05
    floor_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, floor_h / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -floor_h / 2),
    )
    geometries.append(TerrainGeometry(geom=floor_geom, color=(0.45, 0.45, 0.45, 1.0)))

    # Outer border walls
    border_rgba = darken_rgba(brand_ramp(_MUJOCO_BLUE, 0.3), 0.8)
    if self.border_width > 0.0:
      border_boxes = make_border(
        body,
        self.size,
        (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width),
        wall_h,
        (self.size[0] / 2, self.size[1] / 2, wall_h / 2),
      )
      for b_geom in border_boxes:
        geometries.append(TerrainGeometry(geom=b_geom, color=border_rgba))

    # Inner corridors along X axis
    inner_y = self.size[1] - 2 * self.border_width
    num_lanes = max(2, int(np.floor(inner_y / (corr_w + self.wall_thickness))))
    actual_spacing = inner_y / num_lanes

    center_x = self.size[0] / 2
    center_y = self.size[1] / 2
    plat_half = self.platform_width / 2

    wall_color = brand_ramp(_MUJOCO_BLUE, 0.6 + 0.3 * difficulty)

    for i in range(1, num_lanes):
      y_pos = self.border_width + i * actual_spacing

      # Leave center platform clear
      if abs(y_pos - center_y) < (plat_half * 0.7):
        continue

      # Create wall in two segments with a passage gap
      gap_pos_x = self.border_width + rng.uniform(2.0, self.size[0] - 2 * self.border_width - 2.0)
      gap_width = max(1.0, corr_w)

      left_len = gap_pos_x - gap_width / 2 - self.border_width
      right_len = (self.size[0] - self.border_width) - (gap_pos_x + gap_width / 2)

      if left_len > 0.5:
        lx = self.border_width + left_len / 2
        geom = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(np.maximum(1e-6, left_len / 2), np.maximum(1e-6, self.wall_thickness / 2), np.maximum(1e-6, wall_h / 2)),
          pos=(lx, y_pos, wall_h / 2),
        )
        geometries.append(TerrainGeometry(geom=geom, color=wall_color))

      if right_len > 0.5:
        rx = gap_pos_x + gap_width / 2 + right_len / 2
        geom = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(np.maximum(1e-6, right_len / 2), np.maximum(1e-6, self.wall_thickness / 2), np.maximum(1e-6, wall_h / 2)),
          pos=(rx, y_pos, wall_h / 2),
        )
        geometries.append(TerrainGeometry(geom=geom, color=wall_color))

    origin = np.array([center_x, center_y, 0.0])
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class BoxWallsTerrainCfg(SubTerrainCfg):
  """Walls and obstacle barriers (Murs).
  
  Places solid wall barriers and obstacles in alternating orientations,
  requiring the robot to navigate and weave around obstacles.
  """
  num_walls_range: tuple[int, int] = (6, 14)
  wall_length_range: tuple[float, float] = (1.5, 3.5)
  wall_height_range: tuple[float, float] = (0.5, 1.2)
  wall_thickness: float = 0.2
  platform_width: float = 1.6
  border_width: float = 0.25

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []

    # Floor
    floor_h = 0.05
    floor_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, floor_h / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -floor_h / 2),
    )
    geometries.append(TerrainGeometry(geom=floor_geom, color=(0.4, 0.4, 0.42, 1.0)))

    num_walls = int(
      self.num_walls_range[0] + difficulty * (self.num_walls_range[1] - self.num_walls_range[0])
    )
    max_h = self.wall_height_range[1]

    # Border
    border_rgba = darken_rgba(brand_ramp(_MUJOCO_BLUE, 0.1), 0.8)
    if self.border_width > 0.0:
      border_boxes = make_border(
        body,
        self.size,
        (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width),
        max_h,
        (self.size[0] / 2, self.size[1] / 2, max_h / 2),
      )
      for b_geom in border_boxes:
        geometries.append(TerrainGeometry(geom=b_geom, color=border_rgba))

    center_x = self.size[0] / 2
    center_y = self.size[1] / 2
    plat_half = self.platform_width / 2

    for _ in range(num_walls):
      wall_len = rng.uniform(*self.wall_length_range)
      wall_h = rng.uniform(*self.wall_height_range)
      # Orient parallel or perpendicular to X axis
      horizontal = rng.choice([True, False])

      if horizontal:
        sx, sy = wall_len, self.wall_thickness
      else:
        sx, sy = self.wall_thickness, wall_len

      px = rng.uniform(self.border_width + sx / 2 + 0.2, self.size[0] - self.border_width - sx / 2 - 0.2)
      py = rng.uniform(self.border_width + sy / 2 + 0.2, self.size[1] - self.border_width - sy / 2 - 0.2)

      # Avoid center spawn platform
      if (center_x - plat_half - sx / 2 <= px <= center_x + plat_half + sx / 2) and (
        center_y - plat_half - sy / 2 <= py <= center_y + plat_half + sy / 2
      ):
        continue

      wall_color = brand_ramp(_MUJOCO_GREEN, rng.uniform(0.3, 0.8))
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(np.maximum(1e-6, sx / 2), np.maximum(1e-6, sy / 2), np.maximum(1e-6, wall_h / 2)),
        pos=(px, py, wall_h / 2),
      )
      geometries.append(TerrainGeometry(geom=geom, color=wall_color))

    origin = np.array([center_x, center_y, 0.0])
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class BoxPitsAndGapsTerrainCfg(SubTerrainCfg):
  """Holes, pits and trenches in the ground (Trous).
  
  Creates a ground with multiple trenches and pit holes (floor_depth=2.0),
  requiring the robot to step over gaps or navigate around dangerous holes.
  """
  gap_width_range: tuple[float, float] = (0.2, 0.6)
  platform_width: float = 1.4
  border_width: float = 0.25
  floor_depth: float = 2.0
  num_trenches: int = 4

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []

    # Bottom pit floor (2m deep void)
    deep_floor = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, 0.05),
      pos=(self.size[0] / 2, self.size[1] / 2, -self.floor_depth - 0.05),
    )
    geometries.append(TerrainGeometry(geom=deep_floor, color=(0.1, 0.1, 0.12, 1.0)))

    # Surrounding border
    border_rgba = darken_rgba(brand_ramp(_MUJOCO_BLUE, 0.0), 0.85)
    border_boxes = make_border(
      body,
      self.size,
      (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width),
      self.floor_depth,
      (self.size[0] / 2, self.size[1] / 2, -self.floor_depth / 2),
    )
    for b_geom in border_boxes:
      geometries.append(TerrainGeometry(geom=b_geom, color=border_rgba))

    # Center spawn platform
    center_x = self.size[0] / 2
    center_y = self.size[1] / 2
    plat_half = self.platform_width / 2

    center_platform = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(plat_half, plat_half, self.floor_depth / 2),
      pos=(center_x, center_y, -self.floor_depth / 2),
    )
    geometries.append(TerrainGeometry(geom=center_platform, color=(0.5, 0.5, 0.55, 1.0)))

    # Gap width increases with difficulty
    gap_w = self.gap_width_range[0] + difficulty * (
      self.gap_width_range[1] - self.gap_width_range[0]
    )

    # Divide terrain into walkable slabs separated by trenches
    inner_x = self.size[0] - 2 * self.border_width
    num_slabs = self.num_trenches + 1
    total_gap = self.num_trenches * gap_w
    slab_w = max(0.4, (inner_x - total_gap) / num_slabs)

    slab_color = brand_ramp(_MUJOCO_BLUE, 0.4 + 0.4 * difficulty)

    cur_x = self.border_width + slab_w / 2
    for _ in range(num_slabs):
      # Skip if overlapping center spawn
      if not (center_x - plat_half <= cur_x <= center_x + plat_half):
        slab = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(slab_w / 2, (self.size[1] - 2 * self.border_width) / 2, self.floor_depth / 2),
          pos=(cur_x, center_y, -self.floor_depth / 2),
        )
        geometries.append(TerrainGeometry(geom=slab, color=slab_color))
      cur_x += slab_w + gap_w

    origin = np.array([center_x, center_y, 0.0])
    return TerrainOutput(origin=origin, geometries=geometries)


def get_complex_outdoor_terrain_cfg() -> TerrainGeneratorCfg:
  """Configuration containing all complex outdoor terrain types:
  - Marches (Pyramid stairs, inverted stairs, random stairs)
  - Terrains accidentés (Perlin noise outdoor hills, random rough, wave terrain, tilted grid)
  - Murs (Wall obstacles, barriers)
  - Longs couloirs (Narrow corridors)
  - Trous (Pits, trenches, stepping stones over void, narrow beams over void)
  """
  sub_terrains: dict[str, SubTerrainCfg] = {
    # Baseline
    "flat": BoxFlatTerrainCfg(proportion=0.10, size=(10.0, 10.0)),
    
    # 1. Marches (Stairs)
    "pyramid_stairs": BoxPyramidStairsTerrainCfg(
      proportion=0.10,
      size=(10.0, 10.0),
      step_height_range=(0.04, 0.20),
      step_width=0.30,
      platform_width=2.5,
      holes=False,
    ),
    "pyramid_stairs_inv": BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.08,
      size=(10.0, 10.0),
      step_height_range=(0.04, 0.18),
      step_width=0.30,
      platform_width=2.5,
      holes=False,
    ),
    "random_stairs": BoxRandomStairsTerrainCfg(
      proportion=0.08,
      size=(10.0, 10.0),
      step_width=0.6,
      step_height_range=(0.05, 0.20),
      platform_width=2.0,
    ),

    # 2. Terrains accidentés (Outdoor rugged hills, uneven ground)
    "perlin_noise": HfPerlinNoiseTerrainCfg(
      proportion=0.12,
      size=(10.0, 10.0),
      height_range=(0.1, 0.6),
      octaves=4,
      persistence=0.35,
      scale=6.0,
      resolution=0.05,
    ),
    "random_rough": HfRandomUniformTerrainCfg(
      proportion=0.08,
      size=(10.0, 10.0),
      noise_range=(0.02, 0.12),
      noise_step=0.02,
    ),
    "wave_terrain": HfWaveTerrainCfg(
      proportion=0.06,
      size=(10.0, 10.0),
      amplitude_range=(0.05, 0.25),
      num_waves=4,
    ),
    "tilted_grid": BoxTiltedGridTerrainCfg(
      proportion=0.06,
      size=(10.0, 10.0),
      grid_width=0.8,
      tilt_range_deg=15.0,
      height_range=0.20,
    ),

    # 3. Murs (Walls & obstacles)
    "walls": BoxWallsTerrainCfg(
      proportion=0.10,
      size=(10.0, 10.0),
      num_walls_range=(6, 12),
      wall_height_range=(0.5, 1.0),
      wall_length_range=(1.5, 3.0),
      wall_thickness=0.2,
    ),

    # 4. Longs couloirs (Narrow corridors)
    "corridors": BoxCorridorTerrainCfg(
      proportion=0.10,
      size=(10.0, 10.0),
      corridor_width_range=(1.6, 0.95),
      wall_height_range=(0.7, 1.1),
      wall_thickness=0.15,
    ),

    # 5. Trous (Pits, trenches, stepping stones over void)
    "pits_and_trenches": BoxPitsAndGapsTerrainCfg(
      proportion=0.06,
      size=(10.0, 10.0),
      gap_width_range=(0.2, 0.5),
      floor_depth=2.0,
      num_trenches=4,
    ),
    "stepping_stones": BoxSteppingStonesTerrainCfg(
      proportion=0.06,
      size=(10.0, 10.0),
      stone_size_range=(0.5, 0.9),
      stone_distance_range=(0.15, 0.4),
      stone_height=0.2,
      floor_depth=2.0,
    ),
  }

  return TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    sub_terrains=sub_terrains,
    curriculum=True,
  )


def get_stairs_and_holes_terrain_cfg() -> TerrainGeneratorCfg:
  """Étape 2: Intermédiaire - Escaliers et trous (marches et tranchées)."""
  sub_terrains: dict[str, SubTerrainCfg] = {
    # Base plate pour respirer
    "flat": BoxFlatTerrainCfg(proportion=0.25, size=(10.0, 10.0)),

    # Escaliers montants modérés
    "pyramid_stairs": BoxPyramidStairsTerrainCfg(
      proportion=0.25,
      size=(10.0, 10.0),
      step_height_range=(0.03, 0.12),
      step_width=0.35,
      platform_width=2.5,
      holes=False,
    ),

    # Escaliers descendants modérés
    "pyramid_stairs_inv": BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.20,
      size=(10.0, 10.0),
      step_height_range=(0.03, 0.10),
      step_width=0.35,
      platform_width=2.5,
      holes=False,
    ),

    # Escaliers aléatoires / blocs
    "random_stairs": BoxRandomStairsTerrainCfg(
      proportion=0.15,
      size=(10.0, 10.0),
      step_width=0.6,
      step_height_range=(0.04, 0.12),
      platform_width=2.0,
    ),

    # Trous et tranchées modérées
    "pits_and_trenches": BoxPitsAndGapsTerrainCfg(
      proportion=0.15,
      size=(10.0, 10.0),
      gap_width_range=(0.15, 0.35),
      floor_depth=1.5,
      num_trenches=3,
    ),
  }

  return TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=15.0,
    num_rows=8,
    num_cols=16,
    sub_terrains=sub_terrains,
    curriculum=True,
  )

