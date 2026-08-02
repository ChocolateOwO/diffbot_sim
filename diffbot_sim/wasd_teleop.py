#!/usr/bin/env python3
"""WASD teleop: hold a key to move, release it to stop.

A plain terminal never sends real key-up events, so "release to stop" is
approximated: the OS keyboard auto-repeat keeps re-sending the character
while a key is held down, and this node zeroes the Twist as soon as no
character has arrived for STOP_TIMEOUT seconds. Holding a key feels like
continuous motion; letting go stops it shortly after (one repeat interval).
"""
import sys
import termios
import tty
import select
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

LINEAR_SPEED = 0.6     # m/s
ANGULAR_SPEED = 1.0    # rad/s
STOP_TIMEOUT = 0.20    # s of silence => treat key as released
PUBLISH_RATE = 20.0    # Hz

HELP = f"""
WASD teleop -- hold a key to move, release it to stop
  w = forward     s = backward
  a = turn left   d = turn right
  x = stop        q = quit (Ctrl+C also works)
speed: linear {LINEAR_SPEED:.2f} m/s, angular {ANGULAR_SPEED:.2f} rad/s
"""

KEY_TWIST = {
    'w': (LINEAR_SPEED, 0.0),
    's': (-LINEAR_SPEED, 0.0),
    'a': (0.0, ANGULAR_SPEED),
    'd': (0.0, -ANGULAR_SPEED),
    'x': (0.0, 0.0),
}


def get_key(timeout):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.read(1)
    return None


class WasdTeleop(Node):

    def __init__(self):
        super().__init__('wasd_teleop')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def run(self):
        settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        print(HELP)
        last_key_time = 0.0
        lin, ang = 0.0, 0.0
        try:
            while rclpy.ok():
                key = get_key(1.0 / PUBLISH_RATE)
                now = time.time()
                if key:
                    if key == 'q':
                        break
                    if key in KEY_TWIST:
                        lin, ang = KEY_TWIST[key]
                        last_key_time = now
                elif now - last_key_time > STOP_TIMEOUT and (lin or ang):
                    lin, ang = 0.0, 0.0

                msg = Twist()
                msg.linear.x = lin
                msg.angular.z = ang
                self.pub.publish(msg)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            self.pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = WasdTeleop()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.pub.publish(Twist())
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
