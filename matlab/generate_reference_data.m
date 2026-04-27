function generate_reference_data()
%GENERATE_REFERENCE_DATA Invoke the Python exporter for MATLAB reference CSVs.

root = fileparts(fileparts(mfilename('fullpath')));
config = fullfile(root, 'examples', 'configs', 'ekf_imu_gps_baro.json');
outDir = fullfile(root, 'matlab', 'reference');
cmd = sprintf('cd "%s" && python3 -m fusion_sandbox export-matlab --config "%s" --out "%s"', root, config, outDir);
status = system(cmd);
if status ~= 0
    error('Reference export failed with status %d', status);
end
end
