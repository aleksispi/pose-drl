function out = execute_active_view(env, agent, tracker, stats, greedy, ...
                                   ep, recorder, do_record, cams_to_visit)

global CONFIG

% Keep track of environment's initial camera index
init_cam_idx = env.camera_idx;

% Initialize some things
all_recon_errs = cell(1, tracker.nbr_persons);
traj_len = 0;
action.binary = 0;
elev_angle_max = env.scene().camera_rig_individual.elev_angle_span_half;
agent.set_elev_angle_max(elev_angle_max);
% At initial time-step, set agent angle canvas at azimuth angle 0 and
% initial global elevation angle to indicate to agent where it starts
global_angles_cam = env.global_angles_cam();
agent.update_angle_canvas(0, global_angles_cam(2) ...
							 - env.scene().camera_rig_individual.elev_angle_mean);

% Initialize camera rig canvas
agent.init_rig_canvas(env);

% Begin agent-environment interaction
while ~action.binary && traj_len < CONFIG.agent_max_traj_len

    % Get next state
    pose_idxs = tracker.get_detections();
    person_idxs = tracker.person_idxs;
    state = cell(1, tracker.nbr_persons);

    preds = cell(tracker.nbr_persons, 1);
    for i = 1 : tracker.nbr_persons
    
        % Extract current pose index
        pose_idx = pose_idxs(i);
        
        % Goto current person
        env.goto_person(person_idxs(i));
        
        % Get 2D pose prediction and data blob
        [pred, blob] = env.get_current_predictor(pose_idx);
        if i > 1
            % One blob to rule them all ...
            blob = nan;
        end

        % Update state
        state{i} = env.get_state(pose_idx, blob, pred);
    end
    agent.pose_idxs = [agent.pose_idxs; pose_idxs];
    agent.visited_cams = [agent.visited_cams, env.camera_idx];
   
    % Take next action based on state
    [action, data_in] = agent.take_action(state, greedy, env, person_idxs);
    agent.update_hist(ep, data_in, action)
    
	% Insert prediction error
    if CONFIG.agent_reward_autostop_future_improvement || do_record || traj_len == 0 || (action.binary || traj_len >= CONFIG.agent_max_traj_len - 1)
        [agent_recons_nonan, ~] = agent.get_reconstruction();
        for i = 1 : tracker.nbr_persons
            env.goto_person(person_idxs(i));
            all_recon_errs{i} = [all_recon_errs{i}, env.get_recon_error(...
                                 agent_recons_nonan{i})]; %#ok<*AGROW>
        end
    end

	% Go to sampled camera
    old_cam_idx = env.camera_idx;
    if do_record
        old_frame_annots = cell(1, tracker.nbr_persons);
        gt_pose_idxs = nan(1, tracker.nbr_persons);
        for i = 1 : tracker.nbr_persons
            env.goto_person(person_idxs(i));
            old_frame_annots{i} = env.get_frame_annot();
            [gt_pose_idx, ~] = env.scene().get_gt_detection(env.frame_idx, ...
                                                           env.camera_idx, ...
                                                           person_idxs(i));
			gt_pose_idxs(i) = gt_pose_idx;
        end
        old_img = env.get_current_img();
        old_dets = tracker.get_detection_bboxes(pose_idxs);
        agent_angles_global = env.global_angles_cam();
    end
    
    if numel(cams_to_visit) == 0
        % Viewpoints selection from Pose-DRL
        [azim_angle_global, elev_angle_global] = ...
		env.agent_angles_to_global(action.azim_angle, ...
								   action.elev_angle);
    
        env.goto_cam_mises(azim_angle_global, elev_angle_global);
    elseif traj_len < numel(cams_to_visit)
        % Override viewpoint selection when episode recording a baseline
        global_angles_prev = env.global_angles_cam();
        azim_angle_global_prev = global_angles_prev(1);
        elev_angle_global_prev = global_angles_prev(2);
        env.goto_cam(cams_to_visit(traj_len + 1));
        global_angles = env.global_angles_cam();
        azim_angle_global = global_angles(1);
        action.azim_angle = azim_angle_global - azim_angle_global_prev;
        elev_angle_global = global_angles(2);
        action.elev_angle = elev_angle_global - elev_angle_global_prev;
    end
        
    % Strategy for what to do if visiting same camera twice during TESTING
    if greedy
        if strcmp(CONFIG.agent_eval_same_cam_strategy, 'random')
            while any(agent.visited_cams == env.camera_idx)
                env.goto_cam(randi(env.scene().nbr_cameras))
            end
        elseif strcmp(CONFIG.agent_eval_same_cam_strategy, 'continue')
            if agent.visited_cams == env.camera_idx
                action.binary = 1;
            end
        end
    end
    
    next_cam_angles = env.global_angles_cam();
    if (action.binary || traj_len + 1 == CONFIG.agent_max_traj_len)
        env.goto_cam(old_cam_idx);
    else
        % We actually chose new camera, update tracking
        tracker.next(env.frame_idx, env.camera_idx);
    end

	% Record the episode steps (if final trajectory)
    if do_record
	    recorder.record_step(action.azim_angle, action.elev_angle, azim_angle_global, ...
	                         elev_angle_global, agent_angles_global(1), ...
	                         agent_angles_global(2), next_cam_angles(1), ...
	                         next_cam_angles(2), ...
	                         env.scene().camera_rig.cam_coords(old_cam_idx, :), ...
	                         all_recon_errs, old_frame_annots, old_img, ...
	                         env.scene(), old_cam_idx, env.frame_idx, ...
                             old_dets, pose_idxs, gt_pose_idxs, preds);
    end

    % Collect the camera chosen by the agent
    traj_len = traj_len + 1;
    stats.s('Selected cameras dist.').collect(env.camera_idx);

    % End of step updates
    stats.next_step();
end

% Check whether trajectory ended due to max traj len
force_terminated = (traj_len == CONFIG.agent_max_traj_len) && ~action.binary;

% Reward is that of the person whose final recon errors was highest
reward_input_mat = cell2mat(all_recon_errs');
[~, highest_final_recon_idx] = max(reward_input_mat(:, end));
rewards = get_reward(all_recon_errs{highest_final_recon_idx}, agent.visited_cams, force_terminated);
agent.update_hist_reward(ep, rewards);
episode_reward = nansum(rewards.mises) + nansum(rewards.binary);

% The output may vary a bit depending on the mode
sum_cum_recon_error = 0;
sum_init_recon_errs = 0;
sum_cum_recon_error_all = zeros(1, traj_len);

for i = 1 : tracker.nbr_persons
    env.goto_person(person_idxs(i));
    all_rec_errs_i = all_recon_errs{i};
    preds_seq = agent.predictions_seq{i};
    sum_init_recon_errs = sum_init_recon_errs + all_rec_errs_i(1);
    pose_reconstruction = agent.helpers.infer_missing_joints(preds_seq(:, :, 1));
    cum_recon_error = env.get_recon_error(pose_reconstruction);
    sum_cum_recon_error = sum_cum_recon_error + cum_recon_error;
    
    % Potentially do it for 1, ..., N-1 too (for speed-boost in eval)
    if CONFIG.evaluation_compute_all_fixations
        for j = 1 : traj_len
            preds_seq_j = agent.predictions_seq_all{j}{i};
            pose_reconstruction = agent.helpers.infer_missing_joints(preds_seq_j(:, :, 1));
            cum_recon_error_j = env.get_recon_error(pose_reconstruction);
            sum_cum_recon_error_all(j) = sum_cum_recon_error_all(j) + cum_recon_error_j;
        end
    end 
end
stats.collect({'Reconstruction error RL', sum_cum_recon_error / tracker.nbr_persons, ...
               'Reward', episode_reward});
stats.collect({'Traj len', traj_len - 1});
stats.collect({'Hist TLen', traj_len - 1});
out = struct('init_cam_idx', init_cam_idx, 'episode_reward', episode_reward, ...
             'traj_len', traj_len, 'recorder', recorder, ...
             'cum_recon_error', sum_cum_recon_error / tracker.nbr_persons, ...
             'init_recon_error', sum_init_recon_errs / tracker.nbr_persons, ...
             'cum_recon_error_all', sum_cum_recon_error_all / tracker.nbr_persons);
end
