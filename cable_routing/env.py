# Import some basic libraries and functions for this tutorial.
from pydrake.geometry import (
      MeshcatVisualizer,
      MeshcatVisualizerParams,
      StartMeshcat,
)

from pydrake.all import (
        AddMultibodyPlantSceneGraph,
        DiagramBuilder,
        JointSliders,
        Parser,
        Simulator,
)

from pathlib import Path

from models import robot_directives, environment_directives
from triad import AddFrameTriadIllustration

from systems import PrintPose

def initialize_simulation(diagram):
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.)
    return simulator

def load_assembly_task():
    meshcat = StartMeshcat()
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.01)
    parser = Parser(plant, scene_graph)
    parser.SetAutoRenaming(True)
    package_map = parser.package_map()
    package_map.Add("cable_routing",str(Path('../').absolute()))
    parser.AddModelsFromString(robot_directives, ".dmd.yaml")
    parser.AddModelsFromString(environment_directives, ".dmd.yaml")

    plant.Finalize()

    ur5e = plant.GetModelInstanceByName("ur5e")
    eef = plant.GetBodyByName("wrist_3_link", ur5e)

    print_pose = builder.AddSystem(PrintPose(eef.index()))
    builder.Connect(
        plant.get_body_poses_output_port(), print_pose.get_input_port()
    )

    # Visualize frames
    AddFrameTriadIllustration(scene_graph=scene_graph,
                            body=plant.GetBodyByName("waypoint_1"))
    AddFrameTriadIllustration(scene_graph=scene_graph,
                            body=plant.GetBodyByName("waypoint_2"))
    AddFrameTriadIllustration(scene_graph=scene_graph,
                            body=plant.GetBodyByName("wrist_3_link"))
    AddFrameTriadIllustration(scene_graph=scene_graph,
                            body=plant.GetBodyByName("wedge_root"))

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, params=MeshcatVisualizerParams(
            show_hydroelastic=True
        ))

    sliders = builder.AddSystem(JointSliders(meshcat, plant))

    diagram = builder.Build()
    diagram.set_name("nist_board_assembly")
    sliders.Run(diagram, None)
    meshcat.DeleteAddedControls()

    simulator = initialize_simulation(diagram)
    visualizer.StartRecording(False)
    simulator.AdvanceTo(2)
    visualizer.PublishRecording()

if __name__ == '__main__':
    load_assembly_task()
