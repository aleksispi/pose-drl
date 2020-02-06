function train_agent()
% Trains a pose agent on the panoptic environment

% Setup global config settings
global CONFIG

% Set random seed
rng(CONFIG.rng_seed);

% Create statistics collector
stats = StatCollector('Training StatCollector');
stats.register('lr', 'Learning Rate');
stats.register('current', 'm params');
stats.register('avg', 'Loss');
stats.register('avg', 'Reconstruction error RL');
if CONFIG.sequence_length_train > 1
    stats.register('avg', 'Recon err RL init frame');
else
    stats.register('noop', 'Recon err RL init frame');
end
stats.register('avg', 'Reward');
stats.register('avg', 'Traj len');
stats.register('hist', 'Hist TLen');
stats.register('hist', 'Selected cameras dist.', struct('sort_bins_by', 'count'));
for i = 1 : numel(CONFIG.training_baselines)
    str = CONFIG.training_baselines{i};
    stats.register('avg', str);
end
stats.register('hist', 'Cam percentiles');

% Set up diary (i.e. make sure Matlab command window also prints to file)
diary(strcat(CONFIG.output_dir, '/training_log.txt'));
copyfile('load_config.m', strcat(CONFIG.output_dir, '/load_config.m'));
copyfile(CONFIG.agent_solver_proto, strcat(CONFIG.output_dir, '/solver.prototxt'));
copyfile(CONFIG.agent_train_proto, strcat(CONFIG.output_dir, '/train.prototxt'));
copyfile(CONFIG.agent_deploy_proto, strcat(CONFIG.output_dir, '/deploy.prototxt'));

% Setup caffe
helper = Helpers();
helper.setup_caffe();
predictor = nan;

% Launch panoptic train environment
env = Panoptic(CONFIG.dataset_path, CONFIG.dataset_cache, predictor);

% Setup panoptic environment for validation part, to be used when
% evaluating snapshots of agent network as we go along
env_val = setup_validation_environment(env);

% Generate time-freeze-camera order for validation set (note that inside
% this function we have a certain random seed, so that we will always get
% same order in this independently of machine or other settings)
env_val_time_freeze_cams = get_env_val_time_freeze_cams(env_val, helper);

% Update max-elev-span parameter in load_config
CONFIG.agent_elev_angle_max = env.elev_angle_span_half;
helper.set_train_proto();

% Setup solver
solver = caffe.get_solver(CONFIG.agent_solver_proto);

% Get agent network
net = solver.net;
net_val = caffe.Net(CONFIG.agent_train_proto, 'test');
if ~CONFIG.agent_random_init
    fprintf('Loaded RL network weights from %s\n', CONFIG.agent_weights);
    net.copy_from(CONFIG.agent_weights);
end

% Only need to register data names which are given / produced both in
% forward and backward directions
data_names = {'data', 'pred', 'aux', 'canvas', 'rig', 'm', ...
              'action', 'elev_mult', 'rewards_mises', 'rewards_binary'};
agent = PoseDRL(net, solver, CONFIG.training_agent_eps_per_batch, ...
                data_names, CONFIG.sequence_length_train);

% Run training
fprintf('===========| Running RL training! |===========\n')
fprintf('Network params: %d\n\n', helper.count_network_params(net));
[snapshot_results, pathname] = run_training(env, agent, stats, env_val, net_val, ...
                                            env_val_time_freeze_cams);

% Save final model
[pathname, save_path] = save_agent_network(agent, pathname);

% Evaluate the agent in deterministic mode on withheld validation set
evaluate_agent_on_validation_set(env_val, net_val, save_path, ...
                                 snapshot_results, pathname, ...
                                 data_names, stats, ...
                                 env_val_time_freeze_cams, agent.last_trained_ep)

% Turn off diary
diary off;
end

function [snapshot_results, pathname, snapshot_errs_vs_steps, ...
          init_cam_idxs_steps] = run_training(env, agent, stats, env_val, ...
                                              net_val, env_val_time_freeze_cams)

    % Load global config settings
    global CONFIG

    % Create episode recorder and baseline comparisons
    recorder = EpisodeRecorder(env, agent);
    baselines_runner = BaselinesRunner(CONFIG.training_baselines);
    snapshot_errs_vs_steps = [];
    init_cam_idxs_steps = [];

    % Reset agent and environment before trainig
    agent.reset();
    env.reset();

    % To train in true "epoch manner", we first create a randomly
    % ordered indexing of all time-freeze-cams over which we will iterate
    helper = Helpers();
    time_freezes = helper.get_time_freezes(env, 'train');
    nbr_time_freezes = size(time_freezes, 1);
    CONFIG.panoptic_nbr_time_freezes = nbr_time_freezes;
    time_freeze_ctr = 1;
    [time_freezes, ~] = shuffle_time_freezes(time_freezes, 1);
    if iscell(time_freezes)
        env.goto_scene(time_freezes{1, 1});
    end
    
    % Create timer
    timer = Timer('RL Training Timer');

    % Keep track of which snapshot weights yield which results on the
    % validation set
    snapshot_results = {};

    % Potentially evaluate initial agent
    if CONFIG.evaluation_init_snapshot
        if ~exist('pathname', 'var')
            [pathname, save_path] = save_agent_network(agent, nan, 0);
        else
            [pathname, save_path] = save_agent_network(agent, pathname, 0); %#ok<*NODEF>
        end
        % Evaluate current set of weights on validation set
        err_std_all = cell(1, numel(CONFIG.evaluation_nbr_actions));
        err_std_bls_all = cell(1, numel(CONFIG.evaluation_nbr_actions));
        err_std_bls_all_traj_len = cell(1, numel(CONFIG.evaluation_nbr_actions));
        err_std_scenes_all = cell(1, numel(CONFIG.evaluation_nbr_actions));
        err_std_bls_scenes_all = cell(1, numel(CONFIG.evaluation_nbr_actions));
        avg_traj_lens_all = cell(1, numel(CONFIG.evaluation_nbr_actions));
        for i = 1 : numel(CONFIG.evaluation_nbr_actions)
            eval_out = run_evaluation(env_val, net_val, save_path, ...
                                      agent.data_names, env_val_time_freeze_cams, ...
                                      CONFIG.evaluation_nbr_actions(i), ...
                                      agent.last_trained_ep);
            err_std_all{i} = eval_out.err_std;
            err_std_bls_all{i} = eval_out.err_std_bls;
            err_std_bls_all_traj_len{i} = eval_out.err_std_bls_traj_len;
            err_std_scenes_all{i} = eval_out.err_std_scenes;
            err_std_bls_scenes_all{i} = eval_out.err_std_bls_scenes;
            avg_traj_lens_all{i} = eval_out.avg_traj_len;
        end
        snapshot_results{end + 1} = {pathname, err_std_all, err_std_bls_all, ...
                                     0, err_std_scenes_all, err_std_bls_scenes_all, ...
                                     avg_traj_lens_all, err_std_bls_all_traj_len}; %#ok<*AGROW>
    end

    % Train for nbr_total_eps episodes
    batch_ctr = 1;
    for ep = 1 : CONFIG.training_agent_nbr_eps

        % Time entire episode
        timer.tic('episode');
        
        if CONFIG.predictor_limited_caching
            error('Limited caching not supported');
        end

		% If an epoch is complete, reshuffle training set and
        % re-initialize training set indexer
        if time_freeze_ctr > nbr_time_freezes
            time_freeze_ctr = 1;
            time_freezes = helper.get_time_freezes(env, 'train');
            nbr_time_freezes = size(time_freezes, 1);
            [time_freezes, ~] = shuffle_time_freezes(time_freezes);
            stats.next_epoch();
        end
        time_freeze = time_freezes(time_freeze_ctr, :);
        time_freeze_ctr = time_freeze_ctr + 1;
        env.reset();
        env.goto_scene(time_freeze(1));
        env.goto_frame(time_freeze(2));
        env.goto_cam(time_freeze(3));
        env.goto_person(time_freeze(4));

        % Run agent-environment interaction (i.e. let Pose-DRL generate an
        % active-sequence in Panoptic)
        timer.tic('rl_forward');
        [out_sequence, person_idxs, recorder] = ...
            execute_active_sequence(env, agent, stats, 0, ep, ...
                                    CONFIG.sequence_length_train, ...
                                    recorder, 0);
        timer.toc('rl_forward');

        % Run baselines
        timer.tic('Baselines');
		bl_errors = baselines_runner.run(env, out_sequence, person_idxs);
        % Collect training statistics
        for i = 1 : size(bl_errors, 1)
            for j = 1 : size(bl_errors, 2)
                stats.s(bl_errors{i}.name).collect([bl_errors{i, j}.error, ...
                        numel(bl_errors{i, j}.visited_cams) - 1]);
            end
        end
        timer.toc('Baselines');

        % Perform train step
        if rem(ep, CONFIG.training_agent_eps_per_batch) == 0

            % Update network params via backprop (policy gradients)
            agent.train_agent(stats, ep);

            % Update m parameters over time
            if batch_ctr < size(CONFIG.agent_m_values, 2) - 1
				batch_ctr = batch_ctr + 1;
				agent.m_azim = CONFIG.agent_m_values(1, batch_ctr);
				agent.m_elev = CONFIG.agent_m_values(2, batch_ctr);
            end
        end
        stats.collect({'m params', [agent.m_azim, agent.m_elev]});

        % Occassionally snapshot current weights
        if ep < CONFIG.training_agent_nbr_eps && rem(ep, CONFIG.training_agent_snapshot_ep) == 0
            if ~exist('pathname', 'var')
                [pathname, save_path] = save_agent_network(agent, nan, ep);
            else
                [pathname, save_path] = save_agent_network(agent, pathname, ep);
            end

            % Evaluate current set of weights on validation set
			err_std_all = cell(1, numel(CONFIG.evaluation_nbr_actions));
			err_std_bls_all = cell(1, numel(CONFIG.evaluation_nbr_actions));
			err_std_bls_all_traj_len = cell(1, numel(CONFIG.evaluation_nbr_actions));
			err_std_scenes_all = cell(1, numel(CONFIG.evaluation_nbr_actions));
			err_std_bls_scenes_all = cell(1, numel(CONFIG.evaluation_nbr_actions));
			avg_traj_lens_all = cell(1, numel(CONFIG.evaluation_nbr_actions));
			for i = 1 : numel(CONFIG.evaluation_nbr_actions)
				eval_out = run_evaluation(env_val, net_val, save_path, ...
                                          agent.data_names, env_val_time_freeze_cams, ...
                                          CONFIG.evaluation_nbr_actions(i), ...
                                          agent.last_trained_ep);
				err_std_all{i} = eval_out.err_std;
				err_std_bls_all{i} = eval_out.err_std_bls;
				err_std_bls_all_traj_len{i} = eval_out.err_std_bls_traj_len;
				err_std_scenes_all{i} = eval_out.err_std_scenes;
				err_std_bls_scenes_all{i} = eval_out.err_std_bls_scenes;
				avg_traj_lens_all{i} = eval_out.avg_traj_len;
			end
			snapshot_results{end + 1} = {pathname, err_std_all, err_std_bls_all, ...
										 ep, err_std_scenes_all, err_std_bls_scenes_all, ...
										 avg_traj_lens_all, err_std_bls_all_traj_len};
            print_snapshot_results(snapshot_results, stats)
        end

        % Occassionally plot statistics
        if rem(ep, CONFIG.stats_print_iter) == 0
            stats.print();
        end
        timer.toc('episode');
        stats.next_ep();
    end

    % Plot total training statistics over training sequence
    try stats.plot(CONFIG.output_dir); catch; end

    % Make sure the pathname variable exists
    if ~exist('pathname', 'var')
        pathname = nan;
    end
end

function eval_out = run_evaluation(env, net, weight_path, data_names, ...
                           env_val_time_freeze_cams, nbr_actions, ...
                           last_trained_ep, is_final_snapshot)

    % Load global config
    global CONFIG

    % Default args
    if ~exist('is_final_snapshot', 'var')
        is_final_snapshot = 0;
    end
    
    % Store original number of fixations, will need to reset back to that
    orig_nbr_actions = CONFIG.agent_nbr_actions;
    CONFIG.agent_nbr_actions = nbr_actions;
    
    % Begin baseline timing
    fprintf('Evaluating agent + baselines on validation set!\n\n')

    % Copy current snapshot's weights
    net.copy_from(weight_path);

    % Setup RL agent
    agent = PoseDRL(net, nan, CONFIG.training_agent_eps_per_batch, ...
                    data_names, CONFIG.sequence_length_eval);
    agent.last_trained_ep = last_trained_ep;

    % Setup stat collector for evaluation
    eval_stats = StatCollector('Evaluation StatCollector');
    eval_stats.register('avg', 'Reconstruction error RL');
    if CONFIG.sequence_length_eval > 1
        eval_stats.register('avg', 'Recon err RL init frame');
    else
        eval_stats.register('noop', 'Recon err RL init frame');
    end
    eval_stats.register('avg', 'Reward');
    eval_stats.register('avg', 'Traj len');
    eval_stats.register('hist', 'Hist TLen');
    eval_stats.register('hist', 'Selected cameras dist.', struct('sort_bins_by', 'count'));
    for i = 1 : length(CONFIG.evaluation_baselines)
        str = CONFIG.evaluation_baselines{i};
        eval_stats.register('avg', str);
    end
    eval_stats.register('hist', 'Cam percentiles');

    % Evaluate
    [curr_err_std, curr_err_std_bl] = evaluate_agent(env, agent, ...
                                        eval_stats, ...
                                        env_val_time_freeze_cams, ...
                                        is_final_snapshot);
    eval_stats.print();
    field_names = fieldnames(eval_stats.stats);
    traj_len_struct_idxs = ~cellfun(@isempty, strfind(field_names, 'Traj_len'));
    traj_len_fields = field_names(traj_len_struct_idxs);

    if is_final_snapshot
        try eval_stats.plot(CONFIG.output_dir); catch; end
    end

    fprintf('Evaluation on validation set done!\n\n');

    % Reset number of fixations to use
    CONFIG.agent_nbr_actions = orig_nbr_actions;
    
    % Format output
    eval_out.err_std = curr_err_std.errs;
    eval_out.err_std_bls = curr_err_std_bl.errs;
    
    eval_out.err_std_all = curr_err_std.errs_all;
    eval_out.err_std_bls_all = curr_err_std_bl.errs_all;
    
    eval_out.err_std_bls_traj_len = curr_err_std_bl.traj_lens;
    eval_out.err_std_scenes = curr_err_std;
    eval_out.err_std_bls_scenes = curr_err_std_bl;
    eval_out.avg_traj_len = eval_stats.stats.(traj_len_fields{1}).class.mean;
end
function [pathname, save_path] = save_agent_network(agent, pathname, ep)
    global CONFIG
    if ~exist('ep', 'var'); ep = CONFIG.training_agent_nbr_eps; end
    if isnan(pathname)
        pathname = CONFIG.output_dir;
    end
    if ~exist(pathname, 'dir'); mkdir(pathname); end
    save_path = strcat(pathname, '/ep_', num2str(ep), '.caffemodel');
    agent.net.save(save_path);
    fprintf('Saved model to: %s\n\n', save_path);
end

function env_val = setup_validation_environment(env)
    global CONFIG
    evaluation_mode = CONFIG.evaluation_mode;
    if ~strcmp(evaluation_mode, CONFIG.train_mode)
        % Val paths change data path
        dataset_path = strrep(CONFIG.dataset_path, CONFIG.train_mode, evaluation_mode);
        dataset_cache = strrep(CONFIG.dataset_cache, CONFIG.train_mode, evaluation_mode);
        % Create validation environment
        env_val = Panoptic(dataset_path, dataset_cache, env.scene().predictor);
    else
        env_val = env;
    end
end

function evaluate_agent_on_validation_set(env_val, net_val, save_path, ...
                                          snapshot_results, pathname, ...
                                          data_names, stats, ...
                                          env_val_time_freeze_cams, ...
                                          last_trained_ep)
    global CONFIG

    % Evaluate vs. #fixations as well
    orig_nbr_actions = CONFIG.evaluation_nbr_actions;
    orig_compute_all = CONFIG.evaluation_compute_all_fixations;
    additional_nbr_actions = orig_nbr_actions;
    CONFIG.evaluation_nbr_actions = additional_nbr_actions;

    err_std_all = cell(1, numel(CONFIG.evaluation_nbr_actions));
    err_std_bls_all = cell(1, numel(CONFIG.evaluation_nbr_actions));
    err_std_all_all = cell(1, numel(CONFIG.evaluation_nbr_actions));
    err_std_bls_all_all = cell(1, numel(CONFIG.evaluation_nbr_actions));
    err_std_bls_all_traj_len = cell(1, numel(CONFIG.evaluation_nbr_actions));
    err_std_scenes_all = cell(1, numel(CONFIG.evaluation_nbr_actions));
    err_std_bls_scenes_all = cell(1, numel(CONFIG.evaluation_nbr_actions));
    avg_traj_lens_all = cell(1, numel(CONFIG.evaluation_nbr_actions));
    snapshot_results{end + 1} = nan;
    for i = 1 : numel(CONFIG.evaluation_nbr_actions)
        CONFIG.evaluation_compute_all_fixations = orig_compute_all;
        if any(CONFIG.evaluation_nbr_actions == 0)
            is_final_snapshot = CONFIG.evaluation_nbr_actions(i) == 0;
        else
            is_final_snapshot = i == 1;
        end
        if CONFIG.evaluation_nbr_actions(i) == 0
            CONFIG.evaluation_compute_all_fixations = 0;
        end
        eval_out = run_evaluation(env_val, net_val, save_path, data_names, ...
                                  env_val_time_freeze_cams, ...
                                  CONFIG.evaluation_nbr_actions(i), ...
                                  last_trained_ep, is_final_snapshot);
        err_std_all{i} = eval_out.err_std;
        err_std_bls_all{i} = eval_out.err_std_bls;

        err_std_all_all{i} = eval_out.err_std_all;
        err_std_bls_all_all{i} = eval_out.err_std_bls_all;
        
        err_std_bls_all_traj_len{i} = eval_out.err_std_bls_traj_len;
        err_std_scenes_all{i} = eval_out.err_std_scenes;
        err_std_bls_scenes_all{i} = eval_out.err_std_bls_scenes;
        avg_traj_lens_all{i} = eval_out.avg_traj_len;
		snapshot_results{end} = {pathname, err_std_all, err_std_bls_all, ...
								 CONFIG.training_agent_nbr_eps, ...
								 err_std_scenes_all, err_std_bls_scenes_all, ...
								 avg_traj_lens_all, err_std_bls_all_traj_len, ...
                                 err_std_all_all, err_std_bls_all_all};
		save(strcat(CONFIG.output_dir, 'snapshot_results.mat'), 'snapshot_results');
    end
    CONFIG.evaluation_nbr_actions = orig_nbr_actions;
    print_snapshot_results(snapshot_results, stats, additional_nbr_actions);
end

function print_snapshot_results(snapshot_results, training_stats, tmp_nbr_actions)
    global CONFIG

    if ~exist('tmp_nbr_actions', 'var')
        tmp_nbr_actions = nan;
    end

    fprintf('\nPrinting snapshot results for validation set!\n\n');
    best_loss = inf;
    best_ep = nan;
    x = nan(1, numel(snapshot_results));
    y_agent = nan(numel(CONFIG.evaluation_nbr_actions), numel(snapshot_results));
    y_baselines = nan(numel(CONFIG.evaluation_nbr_actions), numel(CONFIG.evaluation_baselines), numel(snapshot_results));
    y_agent_init = nan(size(y_agent));
    y_baselines_init = nan(size(y_baselines));
    y_baselines_traj_len = nan(size(y_baselines));
    
    x_vs_fix = nan(1, numel(tmp_nbr_actions));
    y_agent_vs_fix = nan(numel(tmp_nbr_actions), 1);
    y_baselines_vs_fix = nan(numel(tmp_nbr_actions), numel(CONFIG.evaluation_baselines));

    mean_traj_length = nan(1, numel(snapshot_results));
    for i = 1 : numel(snapshot_results)
        result = snapshot_results{i};
        fprintf('Weights snapshot number: %d\n', i);

        for nbr_fix_idx = 1 : numel(CONFIG.evaluation_nbr_actions)
            nbr_actions = CONFIG.evaluation_nbr_actions(nbr_fix_idx);
            %if nbr_actions == 0
            err_std = result{2}{nbr_fix_idx};
            bls = result{3}{nbr_fix_idx};
            ep = result{4};
            err_std_scenes = result{5}{nbr_fix_idx};
            err_std_bl_scenes = result{6}{nbr_fix_idx};
            avg_traj_len = result{7}{nbr_fix_idx};
            bls_traj_len = result{8}{nbr_fix_idx};
            if nbr_fix_idx == 1
                % Winner is evaluated only at the first entry (4 fix by
                % default, see load_config)
                loss = err_std(1);
                if loss < best_loss
                    best_loss = loss;
                    best_ep = ep;
                end
            end
            x(i) = ep;
            y_agent(nbr_fix_idx, i) = err_std(1);
            if nbr_actions == 0
                nbr_actions = avg_traj_len;
                mean_traj_length(i) = avg_traj_len;
            end
            fprintf('Agent -- #views: %.3f, mean error: %.3f\n', nbr_actions, err_std(1));
            for j = 1 : numel(CONFIG.evaluation_baselines)
                fprintf('Baseline %s -- #views: %.3f, mean error: %.3f\n', ...
                        CONFIG.evaluation_baselines{j}, nbr_actions, bls(1, j));
                y_baselines(nbr_fix_idx, j, i) = bls(1, j); %#ok<*SAGROW>
            end
            fprintf('\n');

            fprintf('Corresponding init-frame results:\n');
            err_std_init = err_std_scenes.init;
            y_agent_init(nbr_fix_idx, i) = err_std_init(1);
            bls_init = err_std_bl_scenes.init;
            fprintf('Agent -- mean error: %.3f\n', err_std_init(1));
            for j = 1 : numel(CONFIG.evaluation_baselines)
                fprintf('Baseline %s -- mean error: %.3f\n', ...
                        CONFIG.evaluation_baselines{j}, bls_init(1, j));
                y_baselines_init(nbr_fix_idx, j, i) = bls_init(1, j);
            end
            fprintf('\n');

            for j = 1 : numel(CONFIG.evaluation_baselines)
                y_baselines_traj_len(nbr_fix_idx, j, i) = bls_traj_len(1, j);
            end

            fprintf('Results for different types of scenes:\n');
            err_std_pose = err_std_scenes.pose;
            err_std_mafia = err_std_scenes.mafia;
            err_std_ultimatum = err_std_scenes.ultimatum;
            bls_pose = err_std_bl_scenes.pose;
            bls_mafia = err_std_bl_scenes.mafia;
            bls_ultimatum = err_std_bl_scenes.ultimatum;
            if ~any(isnan(err_std_pose))
                fprintf('\nPose:\n');
                fprintf('Agent -- mean error: %.3f\n', err_std_pose(1));
                for j = 1 : numel(CONFIG.evaluation_baselines)
                    fprintf('Baseline %s -- mean error: %.3f\n', ...
                            CONFIG.evaluation_baselines{j}, bls_pose(1, j));
                end
                fprintf('\n');
            end
            if ~any(isnan(err_std_mafia))
                fprintf('\nMafia:\n');
                fprintf('Agent -- mean error: %.3f\n', err_std_mafia(1));
                for j = 1 : numel(CONFIG.evaluation_baselines)
                    fprintf('Baseline %s -- mean error: %.3f\n', ...
                            CONFIG.evaluation_baselines{j}, bls_mafia(1, j));
                end
                fprintf('\n');
            end
            if ~any(isnan(err_std_ultimatum))
                fprintf('\nUltimatum:\n');
                fprintf('Agent -- mean error: %.3f\n', err_std_ultimatum(1));
                for j = 1 : numel(CONFIG.evaluation_baselines)
                    fprintf('Baseline %s -- mean error: %.3f\n', ...
                            CONFIG.evaluation_baselines{j}, bls_ultimatum(1, j));
                end
                fprintf('\n');
            end
        end

        if i == numel(snapshot_results) && CONFIG.evaluation_compute_all_fixations

            % In final snapshot, we also wish to display results vs
            % number of fixations
            err_std_all = result{end - 1};
            err_std_bl_all = result{end};
            for nbr_fix_idx = 1 : max(tmp_nbr_actions) + numel(tmp_nbr_actions) - 1
                if numel(tmp_nbr_actions) > 1
                    nbr_actions = nbr_fix_idx - 1;
                else
                    nbr_actions = nbr_fix_idx;
                end
                if numel(err_std_all) == 1 || nbr_actions == 0
                    err_std_all_curr = err_std_all{1};
                    err_std_bl_all_curr = err_std_bl_all{1};
                elseif numel(err_std_all) > 1
                    err_std_all_curr = err_std_all{2};
                    err_std_bl_all_curr = err_std_bl_all{2};
                end
                err_std = err_std_all_curr(:, nbr_fix_idx);
                bls = err_std_bl_all_curr(:, nbr_fix_idx);
                if nbr_actions == 0
                    avg_traj_len = mean_traj_length(end);
                else
                    avg_traj_len = nbr_actions;
                end
                x_vs_fix(nbr_fix_idx) = avg_traj_len;
                y_agent_vs_fix(nbr_fix_idx) = err_std(1);
                fprintf('Agent -- #views: %.3f, mean error: %.3f\n', nbr_actions, err_std(1));
                for j = 1 : numel(CONFIG.evaluation_baselines)
                    fprintf('Baseline %s -- #views: %.3f, mean error: %.3f\n', ...
                            CONFIG.evaluation_baselines{j}, nbr_actions, bls(j));
                    y_baselines_vs_fix(nbr_fix_idx, j) = bls(j); %#ok<*SAGROW>
                end
                fprintf('\n');
            end
        end
    end

    helpers = Helpers();
    % Create Hyperdock plots
    % Evaluation plots
    plots = [];

    for nbr_fix_idx = 1 : numel(CONFIG.evaluation_nbr_actions)
        nbr_actions = CONFIG.evaluation_nbr_actions(nbr_fix_idx);
        y_tmp = y_agent(nbr_fix_idx, :);
        series = helpers.hyperdock_serie('Agent', x, y_tmp(:)');
        for j = 1 : numel(CONFIG.evaluation_baselines)
            data = y_baselines(nbr_fix_idx, j, :);
            series = [series, helpers.hyperdock_serie(CONFIG.evaluation_baselines{j}, x, data(:)')];
        end
        plot_name = sprintf('Reconstruction Error (%d views)', nbr_actions);
        plots = [plots, helpers.hyperdock_plot(plot_name, 'Episodes', 'Error', series)];
    end
    
    % Same for init
    for nbr_fix_idx = 1 : numel(CONFIG.evaluation_nbr_actions)
        nbr_actions = CONFIG.evaluation_nbr_actions(nbr_fix_idx);
        y_tmp = y_agent_init(nbr_fix_idx, :);
        series = helpers.hyperdock_serie('Agent', x, y_tmp(:)');
        for j = 1 : numel(CONFIG.evaluation_baselines)
            data = y_baselines_init(nbr_fix_idx, j, :);
            series = [series, helpers.hyperdock_serie(CONFIG.evaluation_baselines{j}, x, data(:)')];
        end
        plot_name = sprintf('First Frame Reconstruction Error (%d views)', nbr_actions);
        plots = [plots, helpers.hyperdock_plot(plot_name, 'Episodes', 'Error', series)];
    end

    % Plot auto-stopping trajectory length
    if ~any(isnan(mean_traj_length))
        series = helpers.hyperdock_serie('Autostop Agent', x, mean_traj_length);
        plot_name = sprintf('Average Trajectory Length');
        plots = [plots, helpers.hyperdock_plot(plot_name, 'Episodes', 'Length', series)];
    end

    % Plot for final snapshot the accuracy vs number of fixations
    if exist('x_vs_fix', 'var')
        [x_vs_fix, fixations_ordering] = sort(x_vs_fix);
        y_agent_vs_fix = y_agent_vs_fix(fixations_ordering);
        y_baselines_vs_fix = y_baselines_vs_fix(fixations_ordering, :);
        series = helpers.hyperdock_serie('Agent', x_vs_fix, y_agent_vs_fix(:)');
        for j = 1 : numel(CONFIG.evaluation_baselines)
            data = y_baselines_vs_fix(:, j);
            series = [series, helpers.hyperdock_serie(CONFIG.evaluation_baselines{j}, x_vs_fix, data(:)')];
        end
        plot_name = 'Reconstruction Error vs # Viewpoints';
        plots = [plots, helpers.hyperdock_plot(plot_name, '# cameras visited', 'Error', series)];
    end

    % Training plots
    if training_stats ~= 0
        training_graph_resolution = 50;
        data = training_stats.s('Reconstruction error RL').get_data();
        if numel(data.mas) > training_graph_resolution
            x_points = 1 : floor(numel(data.mas) / training_graph_resolution) : numel(data.mas);
        else
            x_points = 1 : numel(data.mas);
        end
        y_agent = data.mas(x_points);
        series = helpers.hyperdock_serie('Agent', x_points, y_agent);
        plots = [plots, helpers.hyperdock_plot('Training Reconstruction Error', 'Episodes', 'Error', series)];
    end

    % Save plots
    helpers.write_hyperdock_graph(plots);

    % Prints the last val loss to file for external usage.
    helpers.write_hyperdock_loss(best_loss, best_ep);
end

function [time_freezes, scene_ctr] = shuffle_time_freezes(time_freezes, scene_ctr)
    global CONFIG
    if ~isfield(CONFIG, 'trainset_seed')
        CONFIG.trainset_seed = 0;
    else
        CONFIG.trainset_seed = CONFIG.trainset_seed + 1;
    end
    rng(CONFIG.trainset_seed);

    % Default args
    if ~exist('scene_ctr', 'var')
        scene_ctr = nan;
    end

    if ~CONFIG.predictor_limited_caching
        nbr_time_freezes = size(time_freezes, 1);
        time_freezes = time_freezes(randperm(nbr_time_freezes), :);
    else
        % Transform to cell
        time_freezes_cell = cell(numel(unique(time_freezes(:, 1))), 2);
        for i = 1 : size(time_freezes_cell, 1)
            time_freezes_cell{i, 1} = i;
            time_freezes_cell{i, 2} = time_freezes(time_freezes(:, 1) == i, 2 : end);
        end
        time_freezes = time_freezes_cell;

        % Scene-wise shuffle
        [time_freezes, scene_ctr] = shuffle_time_freezes_cell(time_freezes);
    end
    rng(CONFIG.rng_seed);
end

function [time_freezes, scene_ctr] = shuffle_time_freezes_cell(time_freezes)
    nbr_time_freezes = size(time_freezes, 1);
    time_freezes = time_freezes(randperm(nbr_time_freezes), :);
    for i = 1 : nbr_time_freezes
        time_freezes_i = time_freezes{i, 2};
        time_freezes{i, 2} = time_freezes_i(randperm(size(time_freezes_i, 1)), :);
    end
    scene_ctr = 1;
end

function env_val_time_freeze_cam_people = get_env_val_time_freeze_cams(env_val, helper)
    global CONFIG
    rng(0);
    env_val_time_freeze_cams_all = helper.get_time_freezes(env_val, CONFIG.evaluation_mode);
    env_val_time_freeze_cam_people = [];
    nbr_scenes = numel(env_val.scenes);
    for i = 1 : nbr_scenes
        scene_i = env_val_time_freeze_cams_all(env_val_time_freeze_cams_all(:, 1) == i, :);
        unique_frames = unique(scene_i(:, 2));
        nbr_frames = numel(unique_frames);
        for j = 1 : nbr_frames
            time_freeze_ij = scene_i(scene_i(:, 2) == unique_frames(j), :);
            time_freeze_cam_person = time_freeze_ij(randi(size(time_freeze_ij, 1)), :);
            env_val_time_freeze_cam_people = [env_val_time_freeze_cam_people; time_freeze_cam_person];
        end
    end
    rng(CONFIG.rng_seed);
end
