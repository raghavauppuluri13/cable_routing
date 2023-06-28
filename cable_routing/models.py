robot_directives = """
directives:
- add_model:
    name: ur5e
    file: package://cable_routing/models/ur5e_description/urdf/ur5e.sdf
    default_joint_positions:
        shoulder_pan_joint: [0]
        shoulder_lift_joint: [-1.54]
        elbow_joint: [1.2]
        wrist_1_joint: [-1.27]
        wrist_2_joint: [-1.54]
        wrist_3_joint: [0]
- add_model:
    name: wedge
    file: package://cable_routing/models/wedge.sdf
- add_weld:
    parent: ur5e::wrist_3_link
    child: wedge::wedge_root
    X_PC:
        translation: [0, 0.13, 0]
"""

environment_directives = """
directives:
- add_model:
    name: robot_table
    file: package://drake/examples/kuka_iiwa_arm/models/table/extra_heavy_duty_table_surface_only_collision.sdf
- add_weld:
    parent: world
    child: robot_table::link
    X_PC:
        translation: [0, 0, -0.7645]
- add_model:
    name: work_table
    file: package://drake/examples/kuka_iiwa_arm/models/table/extra_heavy_duty_table_surface_only_collision.sdf
- add_weld:
    parent: world
    child: work_table::link
    X_PC:
        translation: [0.75, 0, -0.7645]
- add_model:
    name: nist_board
    file: package://cable_routing/models/nist_board/nist_board.sdf
- add_weld:
    parent: world
    child: nist_board::nist_root
    X_PC:
        translation: [0.25,0.155,0.02]
        rotation: !Rpy { deg: [0, 0, -90] }
- add_weld:
    parent: nist_board::nist_root
    child: nist_board::waypoint_1
    X_PC:
        translation: [0.3743171989917755, 0.04417914152145386, 0.0673823207616806]
- add_weld:
    parent: nist_board::nist_root
    child: nist_board::waypoint_2
    X_PC:
        translation: [0.2403673380613327, 0.1584370881319046, 0.07135289907455444]
"""
