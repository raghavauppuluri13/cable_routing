import pandas as pd
from scipy.spatial.transform import Rotation
import argparse


def read_transformations(csv_file):
    # Read the CSV file
    data = pd.read_csv(csv_file, skiprows=7, header=None)

    # Extract UR5 and board data
    ur5_data = data.iloc[:, 2:9]
    board_data = data.iloc[:, 9:16]

    # Swap Y and Z components
    ur5_data[[3, 6]] = ur5_data[[6, 3]]
    board_data[[3, 6]] = board_data[[6, 3]]

    # Compute the average transformations
    ur5_averages = ur5_data.mean()
    board_averages = board_data.mean()

    return ur5_averages, board_averages


def compute_final_transformation(ur5_averages, board_averages):
    # Convert UR5 and board rotations to Rotation objects
    ur5_rotation = Rotation.from_quat(ur5_averages[:4])
    board_rotation = Rotation.from_quat(board_averages[:4])

    # Calculate the inverse UR5 rotation
    ur5_rotation_inv = ur5_rotation.inv()

    # Calculate the transformation of the board frame expressed in the UR5 frame
    final_rotation = ur5_rotation_inv * board_rotation
    final_position = ur5_rotation_inv.apply(board_averages[4:7] - ur5_averages[4:7])

    # Convert the final rotation to quaternion
    final_rotation_quat = final_rotation.as_quat()

    return final_rotation_quat, final_position


def main():
    parser = argparse.ArgumentParser(
        description="Compute the transformation of the board frame expressed in the UR5 frame."
    )
    parser.add_argument(
        "csv_file",
        help="Path to the CSV file containing the calibration transformations.",
    )
    args = parser.parse_args()

    # Read the transformations from the CSV file
    ur5_averages, board_averages = read_transformations(args.csv_file)

    # Compute the final transformation
    final_rotation_quat, final_position = compute_final_transformation(
        ur5_averages, board_averages
    )

    print("Rotation (Quaternion):", final_rotation_quat)
    print("Position (XYZ):", final_position)


if __name__ == "__main__":
    main()
