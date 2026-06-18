#!/usr/bin/env python3
"""
=============================================================
  CONTROLADOR ROBÔ REAL — GRUPO 2
  UERN — Ciência da Computação — Robótica
=============================================================

DESCRIÇÃO:
  Lê o waypoints.txt gerado pelo dijkstra_offline.py
  e move o Pioneer físico pelo caminho planejado.

PRÉ-REQUISITOS (terminais separados):
  1. roscore
  2. sudo chmod a+wr /dev/ttyUSB0
     rosrun rosaria RosAria

USO:
  python3 controlador_real.py
"""

import math
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

# =============================================================
#  CONFIGURAÇÕES — AJUSTE AQUI
# =============================================================

WAYPOINTS_FILE = "waypoints.txt"   # gerado pelo dijkstra_offline.py
GOAL_TOLERANCE = 0.20              # um pouco maior que no Gazebo (robô real deriva mais)
LINEAR_SPEED   = 0.2               # mais devagar no robô real por segurança
ANGULAR_GAIN   = 1.2               # ganho proporcional angular
RATE_HZ        = 10                # frequência do loop de controle

# =============================================================
#  LEITURA DOS WAYPOINTS
# =============================================================

def load_waypoints(path):
    waypoints = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            x, y = map(float, line.split())
            waypoints.append((x, y))
    print(f"  {len(waypoints)} waypoints carregados de '{path}'")
    return waypoints

# =============================================================
#  CALLBACK DE ODOMETRIA
# =============================================================

pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'recebido': False}

def cb_odom(msg):
    pose['x'] = msg.pose.pose.position.x
    pose['y'] = msg.pose.pose.position.y
    q = msg.pose.pose.orientation
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    pose['yaw'] = yaw
    pose['recebido'] = True

# =============================================================
#  LOOP DE CONTROLE
# =============================================================

def follow_path(waypoints, pub_cmd, rate):
    cmd   = Twist()
    total = len(waypoints)

    print(f"\n  Iniciando navegação com {total} waypoints...")

    for i, (tx, ty) in enumerate(waypoints):
        print(f"  → Waypoint {i+1}/{total}: ({tx:.2f}, {ty:.2f})")

        while not rospy.is_shutdown():
            dx   = tx - pose['x']
            dy   = ty - pose['y']
            dist = math.sqrt(dx**2 + dy**2)

            if dist < GOAL_TOLERANCE:
                print(f"     ✓ Atingido!")
                break

            angle_goal = math.atan2(dy, dx)
            angle_err  = math.atan2(
                math.sin(angle_goal - pose['yaw']),
                math.cos(angle_goal - pose['yaw'])
            )

            cmd.linear.x  = min(LINEAR_SPEED, 0.5 * dist)
            cmd.angular.z = ANGULAR_GAIN * angle_err
            pub_cmd.publish(cmd)
            rate.sleep()

    # Para o robô ao terminar
    cmd.linear.x  = 0.0
    cmd.angular.z = 0.0
    pub_cmd.publish(cmd)
    print("\n  ✓ Destino final atingido!")

# =============================================================
#  MAIN
# =============================================================

def main():
    print("="*55)
    print("  CONTROLADOR ROBÔ REAL — GRUPO 2")
    print("="*55)

    rospy.init_node('controlador_real', anonymous=True)

    # Carrega waypoints
    waypoints = load_waypoints(WAYPOINTS_FILE)

    # Subscriber de odometria
    rospy.Subscriber('/RosAria/pose', Odometry, cb_odom)
    print("  Aguardando odometria do Pioneer...")
    rospy.sleep(2.0)

    if not pose['recebido']:
        rospy.logerr("Odometria não recebida! Verifique se o RosAria está rodando.")
        return

    print(f"  Pioneer conectado — pose inicial: ({pose['x']:.2f}, {pose['y']:.2f})")

    # Publisher de velocidade
    pub_cmd = rospy.Publisher('/RosAria/cmd_vel', Twist, queue_size=10)
    rate    = rospy.Rate(RATE_HZ)

    # Pequena pausa antes de começar
    print("  Iniciando em 3 segundos...")
    rospy.sleep(3.0)

    # Navega
    follow_path(waypoints, pub_cmd, rate)

if __name__ == "__main__":
    main()