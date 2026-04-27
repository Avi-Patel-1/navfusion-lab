function [x, P] = ekf_predict(x, P, imu_accel, dt, accel_noise, bias_walk)
%EKF_PREDICT Constant-acceleration inertial prediction with accelerometer bias.
% x = [px py pz vx vy vz bax bay baz]'

a = imu_accel(:) - x(7:9);
x(1:3) = x(1:3) + x(4:6) * dt + 0.5 * a * dt^2;
x(4:6) = x(4:6) + a * dt;

F = eye(9);
F(1:3, 4:6) = eye(3) * dt;
F(1:3, 7:9) = -0.5 * eye(3) * dt^2;
F(4:6, 7:9) = -eye(3) * dt;

q_acc = accel_noise^2;
q_bias = bias_walk^2;
Q = diag([repmat(0.25 * q_acc * dt^4, 1, 3), ...
          repmat(q_acc * dt^2, 1, 3), ...
          repmat(q_bias * dt, 1, 3)]);
P = F * P * F' + Q;
P = 0.5 * (P + P');
end
