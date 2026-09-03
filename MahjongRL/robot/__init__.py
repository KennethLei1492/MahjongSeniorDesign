"""NexArm integration layer.

Targets the Hiwonder NexArm (advanced kit + leader arm): a LeRobot-compatible
6-DOF arm on Feetech STS-series serial bus servos with a K230 vision module.
Two control paths are provided:

  1. lerobot path (preferred): if the `lerobot` package and the NexArm's
     LeRobot config are installed, ArmController drives the follower arm
     through lerobot's FeetechMotorsBus and taught positions can be recorded
     by physically moving the LEADER arm (robot/record_positions.py).
  2. raw serial path (fallback): servo_bus.py implements the Feetech/SCS
     half-duplex packet protocol directly over pyserial, so the arm works
     even without lerobot installed.

Verify the servo IDs, port name, and joint limits against the shipped
NexArm documentation before first power-on (see robot/config.py).
"""
