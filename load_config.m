function load_config(train_mode)
% options for train_mode: 'train', 'test', 'val'

% Init config
global CONFIG
CONFIG = {};

if ~exist('train_mode', 'var')
    train_mode = 'train';
end
CONFIG.train_mode = train_mode;
CONFIG.always_reset_reconstruction = 0;
CONFIG.sequence_length_train = 10;
CONFIG.sequence_length_eval = 10;

fprintf('CONFIG MODE SET TO: (%s)\n', CONFIG.train_mode);
CONFIG.output_dir = strcat('../rl-results/', datestr(now, 'yyyy-mm-dd_HH.MM.SS'), '/');
mkdir(CONFIG.output_dir);

% Caffe / system parameters
helper = Helpers();
CONFIG.use_gpu = 1;
CONFIG.gpu_id = 0;
CONFIG.rng_seed_caffe = 0;
CONFIG.rng_seed = 0;

CONFIG.caffe_lr_policy = 'multistep';

% ------------------ POSE PREDICTOR PARAMS - START ----------------------------------

CONFIG.muby_weights = 'models/mubynet/LS_iter_230000.caffemodel';
CONFIG.muby_deploy_proto_sampling = 'models/mubynet/deploy_sampling.prototxt';
CONFIG.muby_deploy_proto_scoring = 'models/mubynet/deploy_scoring.prototxt';
CONFIG.muby_cache_path = strcat('../panoptic-cache-hd-final-multi-id/', CONFIG.train_mode, '/mubynet/');

CONFIG.predictor_limited_caching = 0;
CONFIG.predictor_disabled = 1;
CONFIG.predictor_precache = 1; % Force predictor pre-computing on dataset

CONFIG.predictor_collect_blobs_from = {'conv4_4_CPM'};
CONFIG.predictor_use_blob = 'conv4_4_CPM';    

% ------------------ POSE PREDICTOR PARAMS - END ------------------------------------

% ------------------ TRAINING/EVALUATION PARAMS - START -------------------
CONFIG.training_eps_per_scene = 1;
CONFIG.training_agent_snapshot_ep = 5000;
CONFIG.training_agent_nbr_eps = 80000;
CONFIG.training_agent_eps_per_batch = 5;
CONFIG.training_agent_lr = 5e-7;
CONFIG.training_agent_lr_update_factor = 0.5;
CONFIG.training_agent_lr_update_steps = [18000, 36000];
CONFIG.training_agent_batch_size = 40;
CONFIG.training_agent_act_funcs = {'ReLU', 'ReLU', 'TanH', 'TanH', 'TanH', 'TanH'};

CONFIG.agent_ms = {{[1, 10], [25, 50], 1}};
helper.set_agent_m_values();

CONFIG.training_baselines = {};
CONFIG.evaluation_mode = 'val';
CONFIG.evaluation_init_snapshot = 0; % eval at first iteration already? (set to 1 in hdock)
CONFIG.evaluation_nbr_actions = [10]; % Hdock winner decided by first entry
CONFIG.evaluation_compute_all_fixations = 0;

CONFIG.evaluation_baselines = {'random'};

% ------------------ TRAINING/EVALUATION PARAMS - END ---------------------

% ------------------ RL AGENT PARAMS - START ------------------------------

% Agent protos
CONFIG.agent_proto_folder = './models/agent/';
CONFIG.agent_proto_name = 'pose_drl';

CONFIG.agent_solver_proto = helper.set_proto_name('solver');
CONFIG.agent_train_proto = helper.set_proto_name('train');
CONFIG.agent_deploy_proto = CONFIG.agent_train_proto;
CONFIG.agent_random_init = 1; % 1=random init, 0=load agent_weights
CONFIG.agent_weights = '';

CONFIG.agent_nbr_actions = 0; % <= 0: auto stop, >= 1: exactly that many cams
CONFIG.agent_max_traj_len = 10;
CONFIG.agent_step_penalty = -0.0625;
CONFIG.agent_reward_autostop_future_improvement = 0; 
CONFIG.agent_force_stop_penalty = 0;

% Angle canvas and camera rig canvas
CONFIG.agent_use_angle_canvas = 1;
CONFIG.agent_nbr_bins_azim = 9;
CONFIG.agent_nbr_bins_elev = 5;
CONFIG.agent_use_rig_canvas = 1;

% Set what types of pose prediction features to use, and in what quantity
% (0 --> do not use, >= 1 --> use that many of this quantity)
CONFIG.agent_pose_features_global3d = 0;
CONFIG.agent_pose_features_global3d_cum = 0;
CONFIG.agent_pose_features_camera2d = 0;
CONFIG.agent_nbr_stacked_cum_preds = 0;

CONFIG.agent_same_cam_penalty = -2.5;
CONFIG.agent_eval_same_cam_strategy = ''; % '', 'random' or 'continue'

% ------------------ RL AGENT PARAMS - END --------------------------------

% ------------------ DATASET/PANOPTIC PARAMS - START ----------------------
CONFIG.panoptic_joint_mapping = [9, 11, 1, 12, 13, 14, 5, 6, 7, 15, 16, 17, 2, 3, 4];
CONFIG.panoptic_scaling_factor = 0.1; % The ratio between cm and mm

CONFIG.panoptic_scene_filter_train = {}; % {'pose'} --> exclude all "pose" scenes
CONFIG.panoptic_scene_filter_eval = {};

CONFIG.panoptic_match_cost_thresh = 0.5;
CONFIG.panoptic_instance_cache_suffix = 'instance_tuned.mat';
CONFIG.panoptic_instance_cache_suffix_det = '_instance_40k_tuned_2k.mat';
CONFIG.panoptic_detections_name = 'matches_epi_empty';
CONFIG.panoptic_det_muby_min_iou = 0.1;
CONFIG.panoptic_min_iou = 0.4;
CONFIG.panoptic_nbr_appearance_samples = 10;
CONFIG.appearance_gt_iou_occluded = 0.75; % if 1, never occluded

CONFIG.dataset_video_format = 'hd';
CONFIG.dataset_path = strcat('../panoptic-data-hd-final/', CONFIG.train_mode, '/');
CONFIG.dataset_cache = strcat('../panoptic-cache-hd-final-multi-id/', CONFIG.train_mode, '/');        
CONFIG.dataset_img_dimensions = [1080, 1920]; % row, cols
CONFIG.panoptic_crop_margin = [50, 75]; % x, y
CONFIG.agent_elev_angle_max = 0.529450481839650; % can predict elev \in [-elev_max, elev_max]

CONFIG.panoptic_error_type = '3d';

% ------------------ DATASET/PANOPTIC PARAMS - END ------------------------ 


% ------------------ MISC PARAMS - START ----------------------------------
CONFIG.stats_ma_weight = 0.0002;
CONFIG.stats_defaults_show_plots = 0;
CONFIG.stats_save_plots = 1;
CONFIG.stats_print_iter = 10;
CONFIG.stats_disabled_stats = {};

CONFIG.use_recorder = 0;
CONFIG.recorder_save_video = 1;
CONFIG.recorder_reset_video = 0;
CONFIG.recorder_frame_rate = 1;
CONFIG.recorder_create_smpl = 0;
CONFIG.plot_smpl_is_male = 1;

% 1 -purple, 2 - blue, 3 - green, 4 - yellow, 5 - yellow
CONFIG.plot_smpl_color = helper.color_from_pid(2, 5);

% ------------------ DATASET PREPROCESSING - START ------------------------
% The flags below are constants that are used to balance and preprocess
% the datasets. Changing these is not necessary.
CONFIG.sequence_step = 5; % Divides dataset fps by 5
CONFIG.training_sequence_overlap_pose = 0.5;
CONFIG.training_sequence_overlap_mafia = 0.5;
CONFIG.training_sequence_overlap_ultimatum = 0.5;
CONFIG.evaluation_sequence_overlap = 0.50; % how much overlap in sequences for testing model?
% ------------------ DATASET PREPROCESSING - END --------------------------

% ------------------ MISC PARAMS - END ------------------------------------
helper.set_solver_proto();
helper.set_train_proto();
end
