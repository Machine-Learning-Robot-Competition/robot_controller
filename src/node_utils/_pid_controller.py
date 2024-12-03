class PIDController:
    def __init__(self, kp: float, ki: float, kd: float):
        self._kp = kp  # Proportional gain
        self._ki = ki  # Integral gain
        self._kd = kd  # Derivative gain

        self._integral = 0.0
        self._previous_error = 0.0

    def compute(self, setpoint: float, current_value: float, dt: float) -> float:
        # Calculate error
        error = setpoint - current_value

        # Calculate integral
        self._integral += error * dt

        # Calculate derivative
        derivative = (error - self._previous_error) / dt

        # Compute PID output
        output = self._kp * error + self._ki * self._integral + self._kd * derivative

        # Update previous error
        self.previous_error = error

        return output
