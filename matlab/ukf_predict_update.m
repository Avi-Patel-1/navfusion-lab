function [x, P] = ukf_predict_update(x, P, imu_accel, z, R, measurementFcn, dt, accel_noise, bias_walk)
%UKF_PREDICT_UPDATE Compact unscented predict/update reference.
% measurementFcn must accept a 9x1 state and return an mx1 measurement.

n = numel(x);
alpha = 0.35;
beta = 2;
kappa = 0;
lambda = alpha^2 * (n + kappa) - n;
wm = ones(2*n + 1, 1) / (2 * (n + lambda));
wc = wm;
wm(1) = lambda / (n + lambda);
wc(1) = wm(1) + (1 - alpha^2 + beta);

Sx = chol((n + lambda) * P, 'lower');
X = zeros(n, 2*n + 1);
X(:,1) = x;
for i = 1:n
    X(:, 1+i) = x + Sx(:,i);
    X(:, 1+n+i) = x - Sx(:,i);
end

for i = 1:(2*n + 1)
    a = imu_accel(:) - X(7:9,i);
    X(1:3,i) = X(1:3,i) + X(4:6,i) * dt + 0.5 * a * dt^2;
    X(4:6,i) = X(4:6,i) + a * dt;
end

x = X * wm;
q_acc = accel_noise^2;
q_bias = bias_walk^2;
Q = diag([repmat(0.25 * q_acc * dt^4, 1, 3), repmat(q_acc * dt^2, 1, 3), repmat(q_bias * dt, 1, 3)]);
P = Q;
for i = 1:(2*n + 1)
    dx = X(:,i) - x;
    P = P + wc(i) * (dx * dx');
end

m = numel(z);
Z = zeros(m, 2*n + 1);
for i = 1:(2*n + 1)
    Z(:,i) = measurementFcn(X(:,i));
end
zbar = Z * wm;
Pzz = R;
Pxz = zeros(n, m);
for i = 1:(2*n + 1)
    dz = Z(:,i) - zbar;
    dx = X(:,i) - x;
    Pzz = Pzz + wc(i) * (dz * dz');
    Pxz = Pxz + wc(i) * (dx * dz');
end
K = Pxz / Pzz;
x = x + K * (z(:) - zbar);
P = P - K * Pzz * K';
P = 0.5 * (P + P');
end
