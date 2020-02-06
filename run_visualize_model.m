function env = run_visualize_model(train_mode, agent_weights, scene, frame, camera)
clc;
close all

% Add paths
addpath(genpath('code/'));

% Setup global config settings
load_config(train_mode)

% Create helper
helper = Helpers();
helper.setup_caffe();

global CONFIG

% Use global env if exists to speed up loading
global env

% Launch panoptic environment
if ~isobject(env)
    env = Panoptic(CONFIG.dataset_path, CONFIG.dataset_cache, nan);
end

env.reset();
env.goto_scene(scene);

if ~exist('frame', 'var')
    frame = randi(env.scene().nbr_frames);
    fprintf('Selecting random frame: %d\n', frame);
end

if ~exist('camera', 'var')
    camera = randi(env.scene().nbr_cameras);
    fprintf('Selecting random camera: %d\n', camera);
end
    
% Load agent
helper.set_train_proto();
solver = caffe.get_solver(CONFIG.agent_solver_proto);

% Get agent network
net = solver.net;
if ~isnan(agent_weights)
    fprintf('Loaded RL network weights from %s\n', agent_weights);
    net.copy_from(agent_weights);
end

% Only need to register data names which are given / produced both in
% forward and backward directions
data_names = {'data', 'pred', 'aux', 'canvas', 'rig', 'm', ...
              'action', 'elev_mult', 'rewards_mises', ...
              'rewards_binary', 'rewards_keep_cum', 'rewards_keep_curr', ...
              'is_ratios_mises', 'pdf_qs_mises'};
agent = PoseDRL(net, solver, CONFIG.training_agent_eps_per_batch, ...
                data_names, CONFIG.sequence_length_train);

stats = StatCollector('Dummy Visualization');

% Create episode recorder
recorder = EpisodeRecorder(env, agent);

agent.reset();

env.goto_cam(camera);
env.goto_frame(frame);

% Execute an active sequence
out_sequence = execute_active_sequence(env, agent, stats, 1, ...
                                       agent.last_trained_ep, ...
                                       CONFIG.sequence_length_eval, ...
                                       recorder, 1);
                                  
fprintf('Visualizing %s to %s\nS:%s F:%d C:%d\n', agent_weights, CONFIG.output_dir, env.scene.scene_name, frame,  camera);
recorder.plot(0);
end