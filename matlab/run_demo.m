function run_demo()
%RUN_DEMO Run the baseline Python experiment and export MATLAB reference data.

root = fileparts(fileparts(mfilename('fullpath')));
config = fullfile(root, 'examples', 'configs', 'ekf_imu_gps_baro.json');
outDir = fullfile(root, 'outputs', 'matlab_demo');
cmd = sprintf('cd "%s" && python3 -m fusion_sandbox run --config "%s" --out "%s"', root, config, outDir);
status = system(cmd);
if status ~= 0
    error('Demo run failed with status %d', status);
end
generate_reference_data();
fprintf('Demo outputs: %s\n', outDir);
fprintf('Reference CSVs: %s\n', fullfile(root, 'matlab', 'reference'));
end
