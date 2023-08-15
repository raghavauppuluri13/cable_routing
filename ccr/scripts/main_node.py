import numpy as np
import rospy

from geometry_msgs.msg import PoseStamped

from stochastic_mpc import stochastic_mpc
from ur_control import URControl


if __name__ == "__main__":
    rospy.init_node("mpc_node")

    ur_control = URControl()
    pose_sub = rospy.Subscriber("/gelsight/pose")
