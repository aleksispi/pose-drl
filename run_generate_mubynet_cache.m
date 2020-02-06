function run_generate_mubynet_cache(mode)
% Caches MubyNet computations for Panoptic

if ~exist('mode', 'var')
    mode = 'train';
end

clc;
close all;

% Add paths
addpath(genpath('code/'));

% Setup global config settings
load_config(mode);

% Create helper
helper = Helpers();
helper.setup_caffe();

% Create MubyNet wrapper
mubynet = MubyWrapper();

% Launch panoptic environment and start caching
global CONFIG
env = Panoptic(CONFIG.dataset_path, CONFIG.dataset_cache, nan);

for scene_idx = 1 : numel(env.scenes)
    env.goto_scene(scene_idx);
    fprintf('Scene: %s\n', env.scene().scene_name);
    save_path = strcat(CONFIG.dataset_cache, 'mubynet/', env.scene().scene_name);
    if exist(save_path) %#ok<*EXIST>
        continue;
    end
    mkdir(save_path);
    save_path_blobs = strcat(save_path, '/conv4_4_CPM_small.mat');
    save_path_poses = strcat(save_path, '/poses.mat');
    
    blobs = cell(env.scene().nbr_frames, env.scene().nbr_cameras);
    poses = cell(env.scene().nbr_frames, env.scene().nbr_cameras, 2);
    for frame_idx = 1 : env.scene().nbr_frames
        env.goto_frame(frame_idx);
        fprintf('  frame: %d/%d\n', frame_idx, env.scene().nbr_frames);
        for camera_idx = 1 : env.scene().nbr_cameras
            env.goto_cam(camera_idx);

            % Compute MubyNet output for current image
            [pose2d, pose3d, trans, muby_blobs] = mubynet.predict(env.get_current_img()); %#ok<*ASGLU>
            blobs{frame_idx, camera_idx} = muby_blobs.(CONFIG.predictor_use_blob);
            poses{frame_idx, camera_idx, 1} = pose2d;
            poses{frame_idx, camera_idx, 2} = pose3d;
        end
    end
    warning(''); %#ok<*WLAST>
    save(save_path_blobs, 'blobs');
    [warnMsg, ~] = lastwarn;
    if ~isempty(warnMsg)
        save(save_path_blobs, 'blobs', '-v7.3');
    end
    save(save_path_poses, 'poses'); 
    disp('DONE CACHING!');
end
end
