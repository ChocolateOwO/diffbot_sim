# OGM จาก 2D LiDAR ใน Gazebo (แทน CoppeliaSim) — รหัส 660610829

หุ่นยนต์ diffbot (ตัวถัง 100×100×40 ซม., 4 ล้อ, LiDAR 2D 684 ลำแสง 360°) เดินสำรวจแมพ 30×30 ม. (6 ห้อง) ใน Gazebo Classic 11 + ROS 2 Humble
แล้วนำข้อมูล LiDAR มาสร้าง Occupancy Grid Map

## ไฟล์
| ไฟล์ | หน้าที่ |
|---|---|
| `data.csv` | ข้อมูลที่เก็บจาก Gazebo (1 แถว = 1 ตำแหน่งหุ่น): `x, y, yaw, (ระยะ, มุม) × 684` = 1371 คอลัมน์ ฟอร์แมตเดียวกับ `save_laser_show_pointcloud.py` (ลำที่ไม่ชนอะไรใน 12 ม. เป็น `nan`) |
| `load_laser_show_pointcloud.py` | โหลด `data.csv` เป็น X, Y, TH, sensor_length, sensor_angle แล้วแสดง point cloud (ปรับแกนเป็น ±16 ม. สำหรับแมพ Gazebo) |
| `ocgm_660610829.py` | สร้าง OGM ด้วย Bresenham + log-odds, grid 0.1 ม. (320×320) |
| `ocgm_660610829.csv` | **ผลลัพธ์**: array 2 มิติ ค่า 0 (ว่าง) – 1 (ตัน), 0.5 = ยังไม่เคยสำรวจ; แถวแรก = ขอบบนของแมพ (y สูงสุด), คอลัมน์แรก = x ต่ำสุด (−16 ม.) |
| `ocgm_compare_660610829.png` | OGM เทียบกับ Top View ของฉาก แกน/สเกลเดียวกัน |
| `world_topview.py` | วาด Top View จากไฟล์ `office_world.world` และตรวจว่าเส้นทางไม่ชนสิ่งกีดขวาง |

โค้ดฝั่งหุ่น/ซิม (repo `diffbot_sim`): `lidar_recorder.py` (เก็บข้อมูล), `waypoint_driver.py` + `config/route.yaml` (เดินตามจุดอัตโนมัติ), `launch/mapping.launch.py`

## วิธีรัน
```bash
# 1) เดินสำรวจ + เก็บข้อมูลอัตโนมัติ (หุ่นเกิดที่ประตูทางเข้า จบแล้วปิด Gazebo เอง ~10 นาที)
#    เปิด 2 หน้าต่าง: Gazebo + แผนที่ OGM ที่อัปเดตสดตามหุ่น (จบแล้วเซฟ ocgm_660610829.csv/.png ให้เอง)
ros2 launch diffbot_sim mapping.launch.py
#    ไม่เอาหน้าต่างแผนที่สด: live_map:=false   ไม่เปิด Gazebo GUI: gui:=false

# 2) สร้างแผนที่ (ใช้ venv ที่มี numpy + matplotlib)
~/ocgm_venv/bin/python ocgm_660610829.py            # เซฟ csv + รูป แล้วเปิดหน้าต่างแผนที่ (--no-show = ไม่เปิดหน้าต่าง)
~/ocgm_venv/bin/python ocgm_660610829.py --animate  # ดูการสร้างแผนที่ทีละขั้น
~/ocgm_venv/bin/python load_laser_show_pointcloud.py  # ดู point cloud ที่เก็บไว้
```

## หมายเหตุสำหรับนำเสนอ
- ตำแหน่ง x, y, yaw ใช้ค่าจริงจาก Gazebo (`odometry_source = world`) เทียบเท่า `getObjectPosition` ของ CoppeliaSim
- LiDAR อยู่สูง ~0.69 ม. จึง **ไม่เห็น** หน้าโต๊ะ (สูง 0.73–0.78 ม.), โซฟา, กรวย, จานอาหาร — เห็นเฉพาะขาโต๊ะ, พนักเก้าอี้, คน, ชั้นวาง, รถ, ต้นไม้ และผนัง (ในรูป Top View ด้านขวาวัตถุที่ LiDAR มองไม่เห็นจึงถูกทำให้จางลง)
- ช่องที่ยังเป็นสีเทาคือด้านในของชั้นวาง/ต้นไม้ที่ลำแสงเข้าไปไม่ถึง และพื้นที่นอกอาคาร
