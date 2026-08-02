#!/usr/bin/env python3
"""Generates a 30x30m office world (6 rooms, furniture, people) as Gazebo SDF."""

WALL_H = 2.5
WALL_T = 0.2

models = []  # list of xml strings
counter = {"wall": 0, "table": 0, "chair": 0, "person": 0, "shelf": 0, "plant": 0}


def box_model(name, sx, sy, sz, x, y, z, yaw=0.0, rgba="0.85 0.82 0.78 1", static=True, collide=True):
    coll = f"""
        <collision name="collision">
          <geometry><box><size>{sx} {sy} {sz}</size></geometry></collision></geometry>
        </collision>""" if False else ""
    return f"""
    <model name="{name}">
      <static>{'true' if static else 'false'}</static>
      <pose>{x} {y} {z} 0 0 {yaw}</pose>
      <link name="link">
        {"<collision name='collision'><geometry><box><size>%g %g %g</size></box></geometry></collision>" % (sx, sy, sz) if collide else ""}
        <visual name="visual">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
        </visual>
      </link>
    </model>"""


def cyl_model(name, radius, length, x, y, z, rgba="0.85 0.82 0.78 1", static=True, collide=True):
    return f"""
    <model name="{name}">
      <static>{'true' if static else 'false'}</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="link">
        {"<collision name='collision'><geometry><cylinder><radius>%g</radius><length>%g</length></cylinder></geometry></collision>" % (radius, length) if collide else ""}
        <visual name="visual">
          <geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>
          <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
        </visual>
      </link>
    </model>"""


def sphere_model(name, radius, x, y, z, rgba="0.85 0.82 0.78 1"):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry><sphere><radius>{radius}</radius></sphere></geometry></collision>
        <visual name="visual">
          <geometry><sphere><radius>{radius}</radius></sphere></geometry>
          <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
        </visual>
      </link>
    </model>"""


def wall(cx, cy, length, along="x", name=None):
    counter["wall"] += 1
    name = name or f"wall_{counter['wall']}"
    if along == "x":
        sx, sy = length, WALL_T
    else:
        sx, sy = WALL_T, length
    models.append(box_model(name, sx, sy, WALL_H, cx, cy, WALL_H / 2, rgba="0.78 0.75 0.7 1"))


def table(cx, cy, yaw=0.0):
    counter["table"] += 1
    n = f"table_{counter['table']}"
    top_h = 0.05
    top_z = 0.75
    leg_h = top_z - top_h / 2
    lx, ly = 1.2, 0.7
    models.append(box_model(f"{n}_top", lx, ly, top_h, cx, cy, top_z, yaw, rgba="0.55 0.35 0.2 1"))
    dx, dy = lx / 2 - 0.06, ly / 2 - 0.06
    import math
    for sxx in (-1, 1):
        for syy in (-1, 1):
            lx_off = dx * sxx
            ly_off = dy * syy
            wx = cx + lx_off * math.cos(yaw) - ly_off * math.sin(yaw)
            wy = cy + lx_off * math.sin(yaw) + ly_off * math.cos(yaw)
            models.append(cyl_model(f"{n}_leg_{sxx}_{syy}", 0.03, leg_h, wx, wy, leg_h / 2, rgba="0.3 0.2 0.1 1"))


def chair(cx, cy, yaw=0.0):
    counter["chair"] += 1
    n = f"chair_{counter['chair']}"
    seat_z = 0.45
    models.append(box_model(f"{n}_seat", 0.42, 0.42, 0.05, cx, cy, seat_z, yaw, rgba="0.2 0.3 0.55 1"))
    import math
    back_off_x = -0.19
    bx = cx + back_off_x * math.cos(yaw)
    by = cy + back_off_x * math.sin(yaw)
    models.append(box_model(f"{n}_back", 0.05, 0.42, 0.42, bx, by, seat_z + 0.2, yaw, rgba="0.2 0.3 0.55 1"))
    for sxx in (-1, 1):
        for syy in (-1, 1):
            lx_off = 0.18 * sxx
            ly_off = 0.18 * syy
            wx = cx + lx_off * math.cos(yaw) - ly_off * math.sin(yaw)
            wy = cy + lx_off * math.sin(yaw) + ly_off * math.cos(yaw)
            models.append(cyl_model(f"{n}_leg_{sxx}_{syy}", 0.02, 0.45, wx, wy, 0.225, rgba="0.15 0.1 0.05 1"))


def person(cx, cy, yaw=0.0, shirt="0.7 0.2 0.2 1", skin="0.85 0.7 0.55 1"):
    counter["person"] += 1
    n = f"person_{counter['person']}"
    models.append(cyl_model(f"{n}_body", 0.18, 1.2, cx, cy, 0.6, rgba=shirt))
    models.append(sphere_model(f"{n}_head", 0.12, cx, cy, 1.2 + 0.12, rgba=skin))


def shelf(cx, cy, yaw=0.0):
    counter["shelf"] += 1
    n = f"shelf_{counter['shelf']}"
    models.append(box_model(n, 1.6, 0.4, 1.8, cx, cy, 0.9, yaw, rgba="0.45 0.3 0.2 1"))


def sofa(cx, cy, yaw=0.0):
    n = "sofa_1"
    models.append(box_model(f"{n}_base", 1.8, 0.7, 0.4, cx, cy, 0.2, yaw, rgba="0.4 0.15 0.15 1"))
    import math
    back_off = -0.3
    bx = cx + back_off * math.cos(yaw)
    by = cy + back_off * math.sin(yaw)
    models.append(box_model(f"{n}_back", 1.8, 0.15, 0.6, bx, by, 0.35, yaw, rgba="0.4 0.15 0.15 1"))


def plant(cx, cy):
    counter["plant"] = counter.get("plant", 0) + 1
    n = f"plant_{counter['plant']}"
    models.append(cyl_model(f"{n}_pot", 0.16, 0.28, cx, cy, 0.14, rgba="0.5 0.32 0.2 1"))
    models.append(sphere_model(f"{n}_foliage_1", 0.32, cx, cy, 0.28 + 0.30, rgba="0.15 0.45 0.2 1"))
    models.append(sphere_model(f"{n}_foliage_2", 0.22, cx + 0.12, cy + 0.08, 0.28 + 0.5, rgba="0.2 0.5 0.25 1"))
    models.append(sphere_model(f"{n}_foliage_3", 0.2, cx - 0.1, cy - 0.1, 0.28 + 0.42, rgba="0.18 0.48 0.22 1"))


def cone(cx, cy):
    counter["cone"] = counter.get("cone", 0) + 1
    n = f"cone_{counter['cone']}"
    orange = "0.95 0.4 0.05 1"
    models.append(cyl_model(f"{n}_base", 0.18, 0.04, cx, cy, 0.02, rgba=orange))
    models.append(cyl_model(f"{n}_body", 0.09, 0.32, cx, cy, 0.04 + 0.16, rgba=orange))
    models.append(cyl_model(f"{n}_stripe", 0.1, 0.04, cx, cy, 0.04 + 0.27, rgba="0.95 0.95 0.9 1"))


def food(cx, cy, z=0.8, kind="apple"):
    counter["food"] = counter.get("food", 0) + 1
    n = f"food_{counter['food']}"
    colors = {
        "apple": "0.8 0.1 0.1 1",
        "orange": "0.95 0.55 0.05 1",
        "bread": "0.75 0.55 0.3 1",
    }
    models.append(cyl_model(f"{n}_plate", 0.13, 0.02, cx, cy, z, rgba="0.95 0.95 0.92 1", collide=False))
    if kind == "bread":
        models.append(box_model(f"{n}_item", 0.14, 0.08, 0.06, cx, cy, z + 0.04, rgba=colors[kind], collide=False))
    else:
        models.append(sphere_model(f"{n}_item", 0.05, cx, cy, z + 0.06, rgba=colors.get(kind, colors["apple"])))


def car(cx, cy, yaw=0.0, body_color="0.75 0.1 0.1 1"):
    """Compact car-shaped obstacle (~1.6m long), not full scale, sized to fit indoors."""
    import math
    counter["car"] = counter.get("car", 0) + 1
    n = f"car_{counter['car']}"
    wheel_r, wheel_w = 0.14, 0.08
    body_z = wheel_r + 0.16
    models.append(box_model(f"{n}_body", 1.6, 0.8, 0.32, cx, cy, body_z, yaw, rgba=body_color))
    cabin_off = 0.15
    cx2 = cx + cabin_off * math.cos(yaw)
    cy2 = cy + cabin_off * math.sin(yaw)
    models.append(box_model(f"{n}_cabin", 0.7, 0.7, 0.28, cx2, cy2, body_z + 0.3, yaw, rgba="0.75 0.85 0.9 0.9"))
    for sxx, lx_off in ((-1, -0.55), (1, 0.55)):
        for syy in (-1, 1):
            ly_off = 0.42 * syy
            wx = cx + lx_off * math.cos(yaw) - ly_off * math.sin(yaw)
            wy = cy + lx_off * math.sin(yaw) + ly_off * math.cos(yaw)
            models.append(cyl_model(f"{n}_wheel_{sxx}_{syy}", wheel_r, wheel_w, wx, wy, wheel_r,
                                     rgba="0.05 0.05 0.05 1"))


# ---------------- room grid ----------------
# whole area x:-15..15  y:-15..15 ; corridor y:-CORRIDOR_HALF..CORRIDOR_HALF
DOOR_W = 2.8          # room <-> corridor doorways (was 1.4, too tight to turn through)
ENTRANCE_W = 3.0       # main south entrance
CORRIDOR_HALF = 1.5   # corridor width = 3.0m (was 2.0m, too tight for the robot to turn in)

# outer walls, with a south entrance gap
wall(0, 15, 30, "x", "wall_outer_north")
wall(-15, 0, 30, "y", "wall_outer_west")
wall(15, 0, 30, "y", "wall_outer_east")
# south wall split for entrance at x=0
_e_lo, _e_hi = -ENTRANCE_W / 2, ENTRANCE_W / 2
_e_left_len = _e_lo - (-15)
_e_right_len = 15 - _e_hi
wall((-15 + _e_lo) / 2, -15, _e_left_len, "x", "wall_outer_south_left")
wall((_e_hi + 15) / 2, -15, _e_right_len, "x", "wall_outer_south_right")

# interior vertical walls at x=-5 and x=5, split top/bottom around corridor
_v_len = 15 - CORRIDOR_HALF
_v_c = (15 + CORRIDOR_HALF) / 2
for xw in (-5, 5):
    wall(xw, _v_c, _v_len, "y", f"wall_v_{xw}_top")
    wall(xw, -_v_c, _v_len, "y", f"wall_v_{xw}_bottom")

room_centers_x = [-10, 0, 10]
room_bounds_x = [(-15, -5), (-5, 5), (5, 15)]

# horizontal walls at the corridor edges (room <-> corridor)
for row_y, tag in ((CORRIDOR_HALF, "top"), (-CORRIDOR_HALF, "bottom")):
    for (x0, x1), cx in zip(room_bounds_x, room_centers_x):
        door_lo, door_hi = cx - DOOR_W / 2, cx + DOOR_W / 2
        left_len = door_lo - x0
        right_len = x1 - door_hi
        left_c = (x0 + door_lo) / 2
        right_c = (door_hi + x1) / 2
        wall(left_c, row_y, left_len, "x", f"wall_h_{tag}_{cx}_left")
        wall(right_c, row_y, right_len, "x", f"wall_h_{tag}_{cx}_right")

# ---------------- furniture per room ----------------
# Room 1 (top-left)  -10,8  : living room -> sofa + small table
sofa(-13.5, 12.5, yaw=1.5708)
table(-10, 6, yaw=0.0)
chair(-9.2, 6, yaw=3.14159)
chair(-10.8, 6, yaw=0.0)
plant(-6.5, 3)
cone(-6.5, 13)
car(-13, 7, yaw=1.5708, body_color="0.15 0.35 0.75 1")
food(-10, 6.35, z=0.78, kind="apple")
food(-9.65, 5.65, z=0.78, kind="orange")

# Room 2 (top-mid) 0,8 : meeting room -> big table + 4 chairs
table(0, 8)
chair(-0.9, 8, yaw=1.5708)
chair(0.9, 8, yaw=-1.5708)
chair(0, 7.0, yaw=3.14159)
chair(0, 9.0, yaw=0.0)
person(1.8, 10.5, yaw=0.0, shirt="0.2 0.5 0.3 1")
plant(4, 13)
cone(-4, 3)
car(3.3, 4.5, yaw=-0.4, body_color="0.9 0.9 0.9 1")
food(0.3, 8.3, z=0.78, kind="apple")
food(-0.3, 7.7, z=0.78, kind="bread")

# Room 3 (top-right) 10,8 : office -> desk + chair + shelf
table(10, 10, yaw=1.5708)
chair(10, 8.8, yaw=1.5708)
shelf(13.5, 12.5, yaw=1.5708)
plant(7, 13.5)
cone(7, 4)
car(11.5, 4.5, yaw=0.2, body_color="0.2 0.55 0.25 1")
food(9.6, 10, z=0.78, kind="orange")

# Room 4 (bottom-left) -10,-8 : storage -> shelves
shelf(-13.5, -12.5, yaw=1.5708)
shelf(-13.5, -4.0, yaw=1.5708)
table(-10, -8)
person(-8.5, -6.0, yaw=1.5708, shirt="0.3 0.3 0.7 1")
plant(-7, -3)
cone(-7, -13)
car(-11, -11, yaw=0.5, body_color="0.85 0.65 0.05 1")
food(-10, -7.7, z=0.78, kind="bread")

# Room 5 (bottom-mid) 0,-8 : dining/break room
table(-2, -8)
chair(-2.9, -8, yaw=1.5708)
chair(-1.1, -8, yaw=-1.5708)
table(2.5, -8)
chair(2.5, -6.8, yaw=3.14159)
person(2.5, -9.5, yaw=1.5708, shirt="0.8 0.6 0.1 1")
plant(4, -3)
cone(-4, -13)
# red car removed -- was spawning right next to the robot's entrance start point
food(-2, -7.65, z=0.78, kind="apple")
food(2.5, -7.65, z=0.78, kind="orange")

# Room 6 (bottom-right) 10,-8 : office
table(10, -6, yaw=0.0)
chair(10, -7.0, yaw=0.0)
shelf(13.5, -12.5, yaw=1.5708)
chair(8.7, -12.0, yaw=1.5708)
plant(7, -3)
cone(7, -8)
car(11, -9, yaw=-0.3, body_color="0.3 0.3 0.85 1")
food(10, -5.65, z=0.78, kind="bread")

body = "\n".join(models)

sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="office_world">

    <include>
      <uri>model://sun</uri>
    </include>

    <include>
      <uri>model://ground_plane</uri>
    </include>

    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <scene>
      <ambient>0.6 0.6 0.6 1</ambient>
      <background>0.7 0.8 0.9 1</background>
      <shadows>true</shadows>
    </scene>

    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>26 -26 20 0 0.5 2.356194</pose>
      </camera>
    </gui>

    <!-- ============ walls & furniture (30x30m, 6 rooms) ============ -->
{body}

  </world>
</sdf>
"""

import os
out_path = os.path.join(os.path.dirname(__file__), "..", "worlds", "office_world.world")
with open(out_path, "w") as f:
    f.write(sdf)
print("wrote:", os.path.abspath(out_path))

print("models:", len(models))
print("counters:", counter)
