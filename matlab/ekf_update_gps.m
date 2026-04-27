function [x, P, accepted, nis] = ekf_update_gps(x, P, gps_xyz, gps_noise, gate)
%EKF_UPDATE_GPS Linear Cartesian GPS position update with Joseph covariance.

H = zeros(3, 9);
H(:, 1:3) = eye(3);
R = eye(3) * gps_noise^2;
y = gps_xyz(:) - x(1:3);
S = H * P * H' + R;
nis = y' / S * y;
accepted = nis <= gate;
if accepted
    K = P * H' / S;
    I = eye(9);
    x = x + K * y;
    P = (I - K * H) * P * (I - K * H)' + K * R * K';
    P = 0.5 * (P + P');
end
end
