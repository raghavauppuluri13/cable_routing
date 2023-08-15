from ur_rtde import rtde_control
from geometry_msgs.msg import Twist, PoseStamped


def arr_to_pose_msg(arr):
    pose_msg.pose = PoseStamped()
    pose_msg.pose.position.x = arr[0]
    pose_msg.pose.position.y = arr[1]
    pose_msg.pose.position.z = arr[2]
    pose_msg.pose.orientation.x = arr[3]
    pose_msg.pose.orientation.y = arr[4]
    pose_msg.pose.orientation.z = arr[5]
    pose_msg.pose.orientation.w = arr[6]


class URControl:
    def __init__(self, robot_ip):
        import rospy

        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

        rospy.init_node("ur_control", anonymous=True)
        self.eef_pose_pub = rospy.Publisher("/eef_pose", PoseStamped, queue_size=10)
        self.eef_twist_pub = rospy.Subscriber("/eef_twist", Twist, self.twist_cb)
        self.publish_eef_pose()

    def twist_cb(self, msg):
        self.twist = msg.linear.x
        twist = [
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,
            msg.angular.x,
            msg.angular.y,
            msg.angular.z,
        ]
        self.rtde_c.speedL(twist)

    def publish_eef_pose(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            pose_arr = self.rtde_r.getActualTCPPose()
            self.eef_pose_pub.publish(arr_to_pose_msg(pose_arr))
            rate.sleep()

    def stop(self):
        self.rtde_c.stopScript()


if __name__ == "__main__":
    robot_ip = "192.168.1.1"  # Adjust the IP address of your robot
    control = URControl(robot_ip)
