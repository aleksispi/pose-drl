function [mean_std_errors, mean_std_bl_errors] = ...
            evaluate_agent(env, agent, stats, time_freeze_cams, is_final_snapshot)

global CONFIG

% Episode record a particular baseline? Requires that the CONFIG flag
% CONFIG.evaluation_baselines contains exactly one baseline name.
RECORD_BASELINE = 0;

% Default args
if ~exist('is_final_snapshot', 'var')
    is_final_snapshot = 0;
end

% Create baselines runner
baselines_runner = BaselinesRunner(CONFIG.evaluation_baselines);
bl_errors = {};
bl_errors_init_frame = [];
bl_traj_lens = {};

% Setup EpisodeRecorder
recorder = EpisodeRecorder(env, agent);

% Initialize array of errors
results = {};
results_init_frame = [];
avg_traj_len_errs = [];

frame_step_length = max(1, CONFIG.sequence_length_eval - ...
                           ceil(CONFIG.evaluation_sequence_overlap * ...
                           CONFIG.sequence_length_eval));
for i = 1 : frame_step_length : size(time_freeze_cams, 1)

    % Reset environment and agent for current sequence
    scene_idx = time_freeze_cams(i, 1);
    frame_idx = time_freeze_cams(i, 2);
    camera_idx = time_freeze_cams(i, 3);
    person_idx = time_freeze_cams(i, 4);
    env.goto_scene(scene_idx);
    env.goto_frame(frame_idx);
    env.goto_cam(camera_idx);
    env.goto_person(person_idx);
    agent.reset();

    % Run Pose-DRL to produce active-sequence
    do_record = i == 1 && is_final_snapshot && CONFIG.use_recorder && ~RECORD_BASELINE;
    [out_sequence, person_idxs, ~] = execute_active_sequence(env, agent, stats, 1, ...
                                          agent.last_trained_ep, CONFIG.sequence_length_eval, ...
                                          recorder, do_record);

    % Hacky way to episode record active-sequence of a baseline
    % (below code assumes you only use one baseline, e.g. 'Oracle')
    if RECORD_BASELINE
        recorder.reset();
        do_record = 1;
        bl_errors_curr = baselines_runner.run(env, out_sequence, person_idxs);
        bl_visited_cams = {};
        for j = 1 : numel(bl_errors_curr)
            bl_visited_cams{j} = bl_errors_curr{j}.visited_cams(2 : end);
        end
        [out_sequence, person_idxs, ~] = execute_active_sequence(env, agent, stats, 1, ...
                                              agent.last_trained_ep, CONFIG.sequence_length_eval, ...
                                              recorder, do_record, bl_visited_cams);
    end
    
    % Plot agent trajectory for first evaluation sequence for final snapshot
    if do_record
        recorder.plot(0);
        recorder.reset();
    end
    
    % Insert to global result container
    scene_name = strcat('scene_', env.scene().scene_name);
    if ~isfield(results, scene_name)
        results.(scene_name) = {};
    end
    for j = 1 : numel(out_sequence)
        results.(scene_name){end + 1} = {out_sequence{j}.cum_recon_error, out_sequence{j}.cum_recon_error_all};
        avg_traj_len_errs = [avg_traj_len_errs, ...
                             [out_sequence{j}.init_recon_error; ...
                              out_sequence{j}.traj_len]]; %#ok<*AGROW>
    end
    results_init_frame = [results_init_frame, out_sequence{1}.cum_recon_error];

    % Also run baselines
    bl_errors_curr = baselines_runner.run(env, out_sequence, person_idxs);
    for j = 1 : size(bl_errors_curr, 1)
        for k = 1 : size(bl_errors_curr, 2)
            stats.s(bl_errors_curr{j}.name).collect([bl_errors_curr{j, k}.error, bl_errors_curr{j, k}.error]);
        end
    end
    curr_bl_traj_lens = nan(size(bl_errors_curr));
    bl_fields = {'error', 'errors_all'};
    curr_bl_errors = cell(numel(bl_fields), size(bl_errors_curr, 1), size(bl_errors_curr, 2));
    for l = 1 : numel(bl_fields)
        bl_field = bl_fields{l};
        for j = 1 : size(bl_errors_curr, 1) % iterate over baselines
            for k = 1 : size(bl_errors_curr, 2) % iterate over active-views
                curr_bl_errors{l, j, k} = bl_errors_curr{j, k}.(bl_field);
                if l == 1
                    curr_bl_traj_lens(j, k) = numel(bl_errors_curr{j, k}.visited_cams) - 1;
                end
            end
        end
    end
    if ~isfield(bl_errors, scene_name)
        bl_errors.(scene_name) = [];
        bl_traj_lens.(scene_name) = [];
    end
    for j = 1 : numel(out_sequence)
        bl_errors.(scene_name){end + 1} = {vertcat(curr_bl_errors{1, :, j}), ...
                                           vertcat(curr_bl_errors{2, :, j})};
        bl_traj_lens.(scene_name) = [bl_traj_lens.(scene_name), curr_bl_traj_lens(:, j)]; 
    end
    bl_errors_init_frame = [bl_errors_init_frame; vertcat(curr_bl_errors{1, :, 1})'];
end

% Get also total mean over scenes
all_results.errs = [];
all_results.pose = [];
all_results.mafia = [];
all_results.ultimatum = [];
all_results.errs_all = [];
field_names_results = fieldnames(all_results);
for i = 1 : numel(field_names_results)
    bl_errors_all.(field_names_results{i}) = [];
end
bl_errors_all.traj_lens = [];

field_names = fieldnames(results);
for i = 1 : numel(field_names)
    
    % Convert to appropriate-sized cells
    scene_name = field_names{i};
    results.(scene_name) = vertcat(results.(scene_name){:});
    bl_errors.(scene_name) = vertcat(bl_errors.(scene_name){:});
    
    % Insert to containers
    all_results.errs = [all_results.errs; cell2mat(results.(scene_name)(:, 1))];
    if CONFIG.evaluation_compute_all_fixations
        all_results.errs_all = [all_results.errs_all; cell2mat(results.(scene_name)(:, 2))];
    end
    bl_errors_all.errs = [bl_errors_all.errs; cell2mat(bl_errors.(scene_name)(:, 1)')'];
    if CONFIG.evaluation_compute_all_fixations
        tmp = bl_errors.(scene_name)(:, 2);
        tmp = cat(3, tmp{:});
        bl_errors_all.errs_all = cat(3, bl_errors_all.errs_all, tmp);
    end
    bl_errors_all.traj_lens = [bl_errors_all.traj_lens; bl_traj_lens.(scene_name)'];
    
    % Compute mean + std
    mean_std_errors.(scene_name) = [mean(cell2mat(results.(scene_name)(:, 1))), std(cell2mat(results.(scene_name)(:, 1)))];
    mean_std_bl_errors.(scene_name) = [mean(cell2mat(bl_errors.(scene_name)(:, 1)'), 2)'; std(cell2mat(bl_errors.(scene_name)(:, 1)'), [], 2)'];
    
    % Do stuff on scene-level
    if ~isempty(strfind(scene_name, 'pose'))
        all_results.pose = [all_results.pose; cell2mat(results.(scene_name)(:, 1))];
        bl_errors_all.pose = [bl_errors_all.pose; cell2mat(bl_errors.(scene_name)(:, 1)')'];
    elseif ~isempty(strfind(scene_name, 'mafia'))
        all_results.mafia = [all_results.mafia; cell2mat(results.(scene_name)(:, 1))];
        bl_errors_all.mafia = [bl_errors_all.mafia; cell2mat(bl_errors.(scene_name)(:, 1)')'];
    elseif ~isempty(strfind(scene_name, 'ultimatum'))
        all_results.ultimatum = [all_results.ultimatum; cell2mat(results.(scene_name)(:, 1))];
        bl_errors_all.ultimatum = [bl_errors_all.ultimatum; cell2mat(bl_errors.(scene_name)(:, 1)')'];
    end
end

% Compute means + stds
for i = 1 : numel(field_names_results)
    mean_std_errors.(field_names_results{i}) = [mean(all_results.(field_names_results{i})); ...
                                                std(all_results.(field_names_results{i}))];
    if ismatrix(bl_errors_all.(field_names_results{i}))
        mean_std_bl_errors.(field_names_results{i}) = [mean(bl_errors_all.(field_names_results{i}), 1); ...
                                                       std(bl_errors_all.(field_names_results{i}), 1)];
    else
        mean_std_bl_errors.(field_names_results{i}) = [mean(bl_errors_all.(field_names_results{i}), 3); ...
                                                       std(bl_errors_all.(field_names_results{i}), [], 3)];
    end
end
mean_std_errors.init = [mean(results_init_frame); std(results_init_frame)];
mean_std_bl_errors.init = [mean(bl_errors_init_frame, 1); std(bl_errors_init_frame, 1)];
mean_std_bl_errors.traj_lens = [mean(bl_errors_all.traj_lens, 1); std(bl_errors_all.traj_lens, 1)];

end
