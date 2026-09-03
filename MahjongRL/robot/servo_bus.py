"""Minimal Feetech STS/SCS serial-bus servo driver (raw fallback path).

Implements the half-duplex packet protocol used by the STS3215 servos in
LeRobot-style arms like the NexArm:

  [0xFF 0xFF id length instruction params... checksum]
  checksum = ~(id + length + instruction + sum(params)) & 0xFF

Only the registers needed for pick-and-place are covered:
  0x2A GOAL_POSITION (2 bytes) + GOAL_TIME (2) + GOAL_SPEED (2)
  0x38 PRESENT_POSITION (2 bytes, read)
  0x28 TORQUE_ENABLE (1 byte)

If the `lerobot` package is available prefer arm_controller.LeRobotBackend,
which wraps lerobot's own FeetechMotorsBus instead of this module.
"""
import time

try:
    import serial
except ImportError:  # allow importing for tests without pyserial
    serial = None

INST_WRITE, INST_READ = 0x03, 0x02
REG_TORQUE_ENABLE = 0x28
REG_GOAL_POSITION = 0x2A
REG_PRESENT_POSITION = 0x38


class FeetechBus:
    def __init__(self, port, baudrate=1_000_000, timeout=0.05):
        if serial is None:
            raise RuntimeError("pyserial not installed: pip install pyserial")
        self.ser = serial.Serial(port, baudrate, timeout=timeout)

    def close(self):
        self.ser.close()

    # ---------- packet layer ----------
    def _packet(self, servo_id, instruction, params):
        length = len(params) + 2
        body = [servo_id, length, instruction] + list(params)
        checksum = (~sum(body[0:]) & 0xFF)
        return bytes([0xFF, 0xFF] + body + [checksum])

    def _write(self, servo_id, reg, data):
        pkt = self._packet(servo_id, INST_WRITE, [reg] + list(data))
        self.ser.reset_input_buffer()
        self.ser.write(pkt)

    def _read(self, servo_id, reg, n):
        pkt = self._packet(servo_id, INST_READ, [reg, n])
        self.ser.reset_input_buffer()
        self.ser.write(pkt)
        resp = self.ser.read(6 + n)
        if len(resp) < 6 + n or resp[0] != 0xFF or resp[1] != 0xFF:
            return None
        return resp[5:5 + n]

    # ---------- servo commands ----------
    def torque(self, servo_id, enabled):
        self._write(servo_id, REG_TORQUE_ENABLE, [1 if enabled else 0])

    def move(self, servo_id, position, duration_ms=800, speed=0):
        """Move to position (0..4095) over duration_ms."""
        position = max(0, min(4095, int(position)))
        data = [position & 0xFF, (position >> 8) & 0xFF,
                duration_ms & 0xFF, (duration_ms >> 8) & 0xFF,
                speed & 0xFF, (speed >> 8) & 0xFF]
        self._write(servo_id, REG_GOAL_POSITION, data)

    def read_position(self, servo_id):
        raw = self._read(servo_id, REG_PRESENT_POSITION, 2)
        if raw is None:
            return None
        return raw[0] | (raw[1] << 8)

    def move_all(self, targets, duration_ms=800, settle=True):
        """targets: {servo_id: position}. Moves all then optionally waits."""
        for sid, pos in targets.items():
            self.move(sid, pos, duration_ms)
        if settle:
            time.sleep(duration_ms / 1000.0 + 0.1)
