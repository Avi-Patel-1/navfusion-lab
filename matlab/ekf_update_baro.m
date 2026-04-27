function [x, P, accepted, nis] = ekf_update_baro(x, P, altitude_m, baro_noise, gate)
%EKF_UPDATE_BARO Scalar altitude update.

H = zeros(1, 9);
H(3) = 1;
R = baro_noise^2;
y = altitude_m - x(3);
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
