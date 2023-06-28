import numpy as np
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.all import (
        AbstractValue,
        LeafSystem,
)
class PrintPose(LeafSystem):
    def __init__(self, body_index):
        LeafSystem.__init__(self)
        self._body_index = body_index
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self.DeclareForcedPublishEvent(self.Publish)

    def Publish(self, context):
        pose = self.get_input_port().Eval(context)[self._body_index]
        print(pose)
        print(
            "gripper position (m): "
            + np.array2string(
                pose.translation(),
                formatter={"float": lambda x: "{:3.2f}".format(x)},
            )
        )
        print(
            "gripper roll-pitch-yaw (rad):"
            + np.array2string(
                RollPitchYaw(pose.rotation()).vector(),
                formatter={"float": lambda x: "{:3.2f}".format(x)},
            )
        )
