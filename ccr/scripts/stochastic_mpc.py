from numba import cuda
import numpy as np
import time
import math
import cupy as cp

cp.random.seed(100)

DT = 0.01

# input sizes
m = 20
N = 10000
L = 100
dim_a = 6
dim_s = 5
dim_pose = 3
num_elites = 10

linear_lim = 0.01  # limit for linear velocity 0.02 m/s
angular_lim = 0.25  # limit for angular velocity (30 deg/s)

# gpu
threads_per_block = 1024
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block


@cuda.jit
def apply_model(states, actions, costs, goal):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    for j in range(1, actions.shape[1]):
        # model
        x_c = states[i, j - 1, 0]
        y_c = states[i, j - 1, 1]
        z_c = states[i, j - 1, 2]
        d_x = states[i, j - 1, 3]
        d_y = states[i, j - 1, 4]

        vx = actions[i, j, 0]
        vy = actions[i, j, 1]
        vz = actions[i, j, 2]
        wx = actions[i, j, 3]
        wy = actions[i, j, 4]
        wz = actions[i, j, 5]

        states[i, j, 0] = (
            x_c
            - (1 - d_x) * vx * DT
            + x_c * math.cos(wz * DT)
            - y_c * math.sin(wz * DT)
        )
        states[i, j, 1] = (
            y_c
            - (1 - d_y) * vy * DT
            + y_c * math.sin(wz * DT)
            + y_c * math.cos(wz * DT)
        )
        states[i, j, 2] = z_c - vz * DT
        states[i, j, 3] = d_x + math.cos(wz * DT)
        states[i, j, 4] = d_y + math.sin(wz * DT)
        sqrs = 0.0
        for k in range(dim_pose):
            sqrs += (goal[k] - states[i, j, k]) ** 2
        costs[i] += math.sqrt(sqrs)


def smooth_gauss_clip(x, a, b, mu, std):
    return a + (b - a) * (1 - cp.cos(cp.pi * x / 10)) / 2


def stochastic_mpc(initial_state, goal):
    mu = cp.zeros((L, dim_a))
    std = cp.ones((L, dim_a))

    states = cp.zeros((N, L + 1, dim_s), dtype=np.float32)

    states[:, 0, :] = initial_state

    start_time_gpu = time.time()
    for i in range(m):
        actions = cp.random.normal(mu, std, size=(N, L, dim_a)).astype(np.float32)
        actions[:, :, :3] = smooth_gauss_clip(
            actions[:, :, :3], -linear_lim, linear_lim, mu[:, :3], std[:, :3]
        )  # Clipping vx, vy, vz
        actions[:, :, 3:6] = smooth_gauss_clip(
            actions[:, :, 3:6], -angular_lim, angular_lim, mu[:, 3:6], std[:, 3:6]
        )  # Clipping wx, wy, wz

        """
        actions[:, :, :3] = cp.clip(
            actions[:, :, :3], -linear_lim, linear_lim
        )  # Clipping vx, vy, vz
        actions[:, :, 3:6] = cp.clip(
            actions[:, :, 3:6], -angular_lim, angular_lim
        )  # Clipping wx, wy, wz
        """

        costs = cp.zeros(N, dtype=np.float32)
        apply_model[blocks_per_grid, threads_per_block](states, actions, costs, goal)
        ind = cp.argpartition(costs, num_elites)[num_elites:]
        elites = actions[ind]
        mu = cp.mean(elites, axis=0)
        std = cp.std(elites, axis=0)
    end_time_gpu = time.time()

    gpu_execution_time = end_time_gpu - start_time_gpu

    print("gpu: ", gpu_execution_time, " sec")
    print("HZ ", 1 / gpu_execution_time)

    return actions[cp.argmin(costs[ind])]  # return best action trajectory


if __name__ == "__main__":

    optim_a = stochastic_mpc(cp.array([0.2, 0.3, 0.4, 0, 1]), cp.array([1, 1, 1]))

    print("optim a", optim_a)
