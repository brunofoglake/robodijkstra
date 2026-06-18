#!/usr/bin/env python3
"""
=============================================================
  CONTROLADOR GAZEBO — GRUPO 2
  UERN — Ciência da Computação — Robótica
=============================================================

DESCRIÇÃO:
  Lê o waypoints.txt gerado pelo dijkstra_offline.py,
  publica o caminho no RViz e move o robô no Gazebo.

PRÉ-REQUISITOS (terminais separados):
  Terminal 1 — ROS Core
    roscore

    Terminal 2 — Gazebo com Turtlebot3
    export TURTLEBOT3_MODEL=burger
    roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch

    Terminal 3 — Modelo do robô (RobotModel no RViz)
    export TURTLEBOT3_MODEL=burger
    roslaunch turtlebot3_bringup turtlebot3_remote.launch

    Terminal 4 — Mapa
    rosrun map_server map_server ~/map.yaml

    Terminal 5 — Transform map → odom
    rosrun tf static_transform_publisher 0 0 0 0 0 0 map odom 100

    Terminal 6 — RViz
    rosrun rviz rviz
    Dentro do RViz configura:
    Global Options → Fixed Frame → odom

    Add → RobotModel
    Add → By topic → /odom → Path        ← rastro percorrido
    Add → By topic → /planned_path → Path ← caminho planejado
    Add → Map → /map                      ← mapa

    Terminal 7 — Gera o caminho (Dijkstra)
    cd ~/pastadoprojeto
    python3 dijkstra_offline.py

    Terminal 8 — Move o robô
    cd ~/pastadoprojeto
    python3 controlador_gazebo.py

import math
import rospy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from tf.transformations import euler_from_quaternion

# =============================================================
#  CONFIGURAÇÕES — AJUSTE AQUI
# =============================================================

WAYPOINTS_FILE = "waypoints.txt"   # gerado pelo dijkstra_offline.py
GOAL_TOLERANCE = 0.15              # distância para considerar waypoint atingido (m)
LINEAR_SPEED   = 0.3               # velocidade linear máxima (m/s)
ANGULAR_GAIN   = 1.5               # ganho proporcional angular
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
#  PUBLICAR CAMINHO NO RVIZ
# =============================================================

def publish_path(pub_path, waypoints):
    path_msg = Path()
    path_msg.header.frame_id = 'map'
    path_msg.header.stamp    = rospy.Time.now()
    for wx, wy in waypoints:
        ps = PoseStamped()
        ps.header.frame_id    = 'map'
        ps.pose.position.x    = wx
        ps.pose.position.y    = wy
        ps.pose.orientation.w = 1.0
        path_msg.poses.append(ps)
    pub_path.publish(path_msg)
    print(f"  Caminho publicado em /planned_path")
    print(f"  RViz: Add → Path → tópico /planned_path")

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
    print("  CONTROLADOR GAZEBO — GRUPO 2")
    print("="*55)

    rospy.init_node('controlador_gazebo', anonymous=True)

    # Carrega waypoints
    waypoints = load_waypoints(WAYPOINTS_FILE)

    # Subscribers
    rospy.Subscriber('/odom', Odometry, cb_odom)
    rospy.sleep(1.0)

    if not pose['recebido']:
        rospy.logerr("Nenhuma odometria recebida! Verifique se o Gazebo está rodando.")
        return

    print(f"  Odometria OK — pose inicial: ({pose['x']:.2f}, {pose['y']:.2f})")

    # Publishers
    pub_cmd  = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    pub_path = rospy.Publisher('/planned_path', Path, queue_size=1, latch=True)
    rate     = rospy.Rate(RATE_HZ)

    # Publica caminho no RViz
    publish_path(pub_path, waypoints)

    # Navega
    follow_path(waypoints, pub_cmd, rate)

if __name__ == "__main__":
    main()