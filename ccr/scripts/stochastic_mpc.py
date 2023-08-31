import rospy
from numba import cuda
import quaternion as quat
import numpy as np
import time
import math
import cupy as cp

from geometry_msgs.msg import PoseStamped, Twist

cp.random.seed(100)
cp.set_printoptions(precision=7, suppress=True)

DT = 0.001

# input sizes
m = 20
N = 1000
L = 10
dim_a = 6
dim_s = 8
dim_pose = 3
num_elites = 10

W_dist = 1
W_pose = 1
W_smooth = 0.1
discount = 1


linear_lim = 0.005  # limit for linear velocity 0.02 m/s
angular_lim = 0.025  # limit for angular velocity (30 deg/s)

# gpu
threads_per_block = 1024
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

# inference

alpha = 0.8  # incorporation factor (smaller means faster incorporation)


def test_model(states, actions, costs, goal):
    TEST_N = 2
    for i in range(TEST_N):
        for j in range(1, actions.shape[1]):
            # model
            x_c = states[i, j - 1, 0]
            y_c = states[i, j - 1, 1]
            z_c = states[i, j - 1, 2]
            d_x = states[i, j - 1, 3]
            d_y = states[i, j - 1, 4]
            x_eef = states[i, j - 1, 5]
            y_eef = states[i, j - 1, 6]
            z_eef = states[i, j - 1, 7]

            vx = actions[i, j, 0]
            vy = actions[i, j, 1]
            vz = actions[i, j, 2]
            wx = actions[i, j, 3]
            wy = actions[i, j, 4]
            wz = actions[i, j, 5]

            states[i, j, 0] = (
                -(1 - d_x) * vx * DT + x_c * math.cos(wz * DT) - y_c * math.sin(wz * DT)
            )
            states[i, j, 1] = (
                -(1 - d_y) * vy * DT + x_c * math.sin(wz * DT) + y_c * math.cos(wz * DT)
            )
            states[i, j, 2] = z_c - vz * DT
            states[i, j, 3] = d_x * math.cos(wz * DT) - d_y * math.sin(wz * DT)
            states[i, j, 4] = d_x * math.sin(wz * DT) + d_y * math.cos(wz * DT)
            print("states", states[i, j, :])
            print("wz", x_c * math.cos(wz * DT))
            sqrs = 0.0
            for k in range(dim_pose):
                sqrs += (goal[k] - states[i, j, k]) ** 2
            costs[i] += math.sqrt(sqrs) * W_dist
            # print(math.sqrt(sqrs) * W_dist * discount ** j)
    print("STATES: ", states[:TEST_N, :4])
    print("COSTS: ", costs[:TEST_N])
    print("ACTIONS: ", actions[:TEST_N, :4])
    assert False


@cuda.jit(device=True)
def se3_matrix_exp(p_mag, p_unit, phi_mag, phi_unit):
    pass


@cuda.jit
def apply_model(states, actions, costs, goal):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    for j in range(actions.shape[1]):
        # model
        x_c = states[i, j, 0]
        y_c = states[i, j, 1]
        z_c = states[i, j, 2]
        d_x = states[i, j, 3]
        d_y = states[i, j, 4]

        x_e = states[i, j, 5]
        y_e = states[i, j, 6]
        z_e = states[i, j, 7]
        th_xe = states[i, j, 8]
        th_ye = states[i, j, 9]
        th_ze = states[i, j, 10]

        vx = actions[i, j, 0]
        vy = actions[i, j, 1]
        vz = actions[i, j, 2]
        wx = actions[i, j, 3]
        wy = actions[i, j, 4]
        wz = actions[i, j, 5]

        states[i, j + 1, 0] = (
            -(1 - d_x) * vx * DT + x_c * math.cos(wz * DT) - y_c * math.sin(wz * DT)
        )
        states[i, j + 1, 1] = (
            -(1 - d_y) * vy * DT + x_c * math.sin(wz * DT) + y_c * math.cos(wz * DT)
        )
        states[i, j + 1, 2] = z_c - vz * DT
        states[i, j + 1, 3] = d_x * math.cos(wz * DT) - d_y * math.sin(wz * DT)
        states[i, j + 1, 4] = d_x * math.sin(wz * DT) + d_y * math.cos(wz * DT)
        for k in range(4):
            for l in range(4):
                pass

        C[row, col] = tmp
        sqrs = 0.0
        for k in range(dim_pose):
            sqrs += (goal[k] - states[i, j + 1, k]) ** 2
        costs[i] += math.sqrt(sqrs) * W_dist
        for k in range(dim_a):
            costs[i] += W_smooth * (actions[i, j, k] - actions[i, j - 1, k])


def smooth_gauss_clip(x, a, b, mu, std):
    return a + (b - a) * (1 - cp.cos(cp.pi * x / (linear_lim))) / 2


def stochastic_mpc(initial_state, goal):
    mu = cp.zeros((L, dim_a))
    std = cp.ones((L, dim_a))

    states = cp.zeros((N, L + 1, dim_s), dtype=np.float32)

    states[:, 0, :] = initial_state

    start_time_gpu = time.time()
    actions = cp.zeros((N, L, dim_a))
    actions[:, :, :3] = cp.random.uniform(
        -linear_lim, linear_lim, size=(N, L, 3)
    ).astype(np.float32)
    actions[:, :, 3:6] = cp.random.uniform(
        -angular_lim, angular_lim, size=(N, L, 3)
    ).astype(np.float32)
    for i in range(m):
        costs = cp.zeros(N, dtype=np.float32)
        apply_model[blocks_per_grid, threads_per_block](states, actions, costs, goal)
        # test_model(states, actions, costs, goal)
        ind = cp.argpartition(costs, num_elites)[:num_elites]
        elites = actions[ind]
        mu = cp.mean(elites, axis=0)
        std = cp.std(elites, axis=0)
        actions = cp.random.normal(mu, std, size=(N, L, dim_a)).astype(np.float32)
        actions[:, :, :3] = smooth_gauss_clip(
            actions[:, :, :3], -linear_lim, linear_lim, mu[:, :3], std[:, :3]
        )  # Clipping vx, vy, vz
        actions[:, :, 3:6] = smooth_gauss_clip(
            actions[:, :, 3:6], -angular_lim, angular_lim, mu[:, 3:6], std[:, 3:6]
        )  # Clipping wx, wy, wz
    end_time_gpu = time.time()
    # print(costs[ind])

    gpu_execution_time = end_time_gpu - start_time_gpu

    # print("gpu: ", gpu_execution_time, " sec")
    # print("HZ ", 1 / gpu_execution_time)

    return actions[cp.argmin(costs[ind])]  # return best action trajectory


state = None
initial_state = None
eef_pose = None
initial_eef_pose = None
DIFF_POSE = np.array([0, 0.05, 0])


def gelsight_pose_cb(msg):
    global initial_eef_pose, eef_pose, state, initial_state
    if eef_pose is not None and initial_eef_pose is not None:
        pose = np.eye(4)
        pose[:3, :3] = quat.as_rotation_matrix(
            np.quaternion(
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            )
        )
        pose[:3, 3] = np.array(
            [msg.pose.position.y, msg.pose.position.x, msg.pose.position.z]
        )
        T = np.eye(4)
        T[:3, :3] = quat.as_rotation_matrix(
            quat.from_rotation_vector(np.array([0, 0, np.pi / 2]))
        )
        pose = np.matmul(T, pose)

        rot = quat.as_rotation_vector(quat.from_rotation_matrix(pose[:3, :3]))
        rot = np.asarray(rot)
        d_vec = np.array([np.cos(rot[2]), np.sin(rot[2])])
        d_vec /= np.linalg.norm(d_vec)  # unit v
        d_vec[0] = np.abs(d_vec[0])  # x needs to always be positive

        tmp_state = np.zeros(dim_s)
        tmp_state[:3] = np.asarray(pose[2, :3])
        # tmp_state[:3] += eef_pose[:3] - initial_eef_pose[:3]
        tmp_state[3:5] = d_vec
        state = tmp_state
        if initial_state is None:
            initial_state = state.copy()


def eef_pose_cb(msg):
    global initial_eef_pose, eef_pose
    eef_pose = np.array(
        [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ]
    )
    if initial_eef_pose is None:
        initial_eef_pose = eef_pose.copy()


def action_to_twist(action):
    msg = Twist()
    msg.linear.x = action[0]
    msg.linear.y = action[1]
    msg.linear.z = action[2]
    msg.angular.x = action[3]
    msg.angular.y = action[4]
    msg.angular.z = action[5]
    return msg


class RingBuffer:
    def __init__(self, buffer_size, data_shape):
        self.arr = np.zeros((buffer_size, *data_shape))
        print(self.arr.shape)
        self.i = 0
        self.data_shape = data_shape
        self.buffer_size = buffer_size

    def get_last_n(self, n):
        assert n <= self.buffer_size
        inds = np.arange(0, n) * -1 + (self.i - 1)
        return self.arr[inds]

    def step(self, data):
        assert data.shape == self.data_shape, print(data.shape)
        self.arr[self.i] = data
        self.i = (self.i + 1) % self.buffer_size


if __name__ == "__main__":

    rospy.init_node("mpc_node")

    pose_sub = rospy.Subscriber("/gelsight/diff_pose", PoseStamped, gelsight_pose_cb)
    pose_sub = rospy.Subscriber("/eef_pose", PoseStamped, eef_pose_cb)
    twist_pub = rospy.Publisher("/eef_twist", Twist, queue_size=20)

    r = rospy.Rate(20)

    buffer = RingBuffer(L, (L, dim_a))
    w = np.exp(-alpha * np.arange(L))
    w = w[..., None]

    def shutdown_hook():
        twist_pub.publish(action_to_twist(np.zeros(dim_a)))

    rospy.on_shutdown(shutdown_hook)

    while not rospy.is_shutdown():
        if state is not None and eef_pose is not None:
            local_state = cp.asarray(state)
            goal = cp.asarray(initial_state[:3] + DIFF_POSE)
            print(np.linalg.norm(goal - local_state[:3]))
            buffer.step(stochastic_mpc(local_state, goal).get())
            actions = buffer.get_last_n(L)
            actions = actions[np.arange(L), np.arange(L)]
            action = np.sum(w * actions, axis=0)
            twist_pub.publish(action_to_twist(action))
        r.sleep()
