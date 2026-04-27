function [x, P, accepted, nis] = ekf_update_gps_range_bearing(x, P, gps_xyz, gps_noise, gate)
%EKF_UPDATE_GPS_RANGE_BEARING Range/bearing/altitude update from GPS position.

px = x(1); py = x(2); pz = x(3);
rho = max(sqrt(px^2 + py^2), 1e-6);
h = [rho; atan2(py, px); pz];
z = [sqrt(gps_xyz(1)^2 + gps_xyz(2)^2); atan2(gps_xyz(2), gps_xyz(1)); gps_xyz(3)];

H = zeros(3, 9);
H(1,1) = px / rho;
H(1,2) = py / rho;
H(2,1) = -py / rho^2;
H(2,2) = px / rho^2;
H(3,3) = 1;

R = diag([gps_noise^2, (gps_noise / max(rho, 1))^2, gps_noise^2]);
y = z - h;
y(2) = mod(y(2) + pi, 2*pi) - pi;
S = H * P * H' + R;
nis = y' / S * y;
accepted = nis <= gate;
if accepted
    K = P * H' / S;
    x = x + K * y;
    I = eye(9);
    P = (I - K * H) * P * (I - K * H)' + K * R * K';
    P = 0.5 * (P + P');
end
end
