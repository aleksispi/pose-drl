classdef PoseDRL < handle
    % Pose-DRL agent for active 3d human pose estimation

    properties
        
        solver % solver for updating in backprop (learning rate etc)
        net % policy network
        data_names % names of all inputs / actions / rewards / labels
        data_ndims % keeps track of data sizes
        history % history is cell array with history_size elements
        history_size
        batch_size % batch size for caffe backprop (may differ from m above)
        nbr_actions_taken % number of actions taken in active-view
        nbr_actions_taken_seq % number of actions taken in active-sequence
        sequence_length % length of active-sequence
        
        m_azim % von Mises inverse variance (azimuth angle)
        m_elev % von Mises inverse variance (elevation angle)
        elev_angle_max % elevation angle maximum
        von_mises_dist % von Mises class (for sampling camera locations)
        
        img_dims % used in 2d pose features
        pred_features % pose prediction features
        pred_size % number of features related to person's pose
        nbr_stacked_cum_preds % size of pose prediction history
        
        last_trained_ep % The last ep that train_agent was called for
        frame_starts % Tracks what actions taken over active-sequence (for backprop)
        
        helpers % various utility functions
        
        nbr_actions % number of actions the agent will take
        max_traj_len % max length of active-view
        predictions % 3d pose predictions
        visited_cams % indices of cameras visited
        
        disable_temporal_fusion % omit temporal fusion?
        pose_idxs % person indices from tracking, used to fuse for correct persons
        nbr_dets_all % to measure average number of detections, and auxiliary feature for agent

        % Below fields are for action history (locations on viewing sphere)
        use_canvas % whether to use an angle canvas
        use_rig_canvas % wheter to use a rig canvas
        canvas % discretization of visited angles (azim-elev)
        rig_canvas % discretization of camera rig
        azim_centers % centers needed to assign agent angle to canvas
        elev_centers % centers needed to assign agent angle to canvas
        nbr_bins_azim % granularity of azim angle discretization
        nbr_bins_elev % granularity of elev angle discretization
        prev_azim % relative to starting azim (which is always 0)
        curr_azim_bin

        % cum-preds throughout video sequence
        predictions_seq
        predictions_seq_ok
        predictions_seq_all % keep track also of 1, ..., N - 1 for effciciency
        predictions_seq_ok_all
        compute_all_fix
    end

    methods

        function obj = PoseDRL(net, solver, history_size, data_names, sequence_length)
            % Constructor
            global CONFIG

            % Default args
            if ~exist('sequence_length', 'var')
                sequence_length = CONFIG.sequence_length_train;
            end

            % Set object fields
            obj.net = net;
            obj.solver = solver;
            obj.data_names = data_names;
            obj.data_ndims = {};
            for i = 1 : numel(obj.data_names)
                obj.data_ndims.(obj.data_names{i}) = nan;
            end
            obj.history_size = history_size;
            obj.batch_size = CONFIG.training_agent_batch_size;
            obj.m_azim = CONFIG.agent_m_values(1, 1);
            obj.m_elev = CONFIG.agent_m_values(2, 1);
            obj.elev_angle_max = CONFIG.agent_elev_angle_max;
            obj.von_mises_dist = VonMises();
            obj.img_dims = fliplr(CONFIG.dataset_img_dimensions);
            obj.last_trained_ep = 1;
            obj.frame_starts = [];
            obj.sequence_length = sequence_length;
            obj.helpers = Helpers();
            obj.nbr_actions = CONFIG.agent_nbr_actions;
            obj.max_traj_len = CONFIG.agent_max_traj_len;
            obj.pred_features.camera2d = CONFIG.agent_pose_features_camera2d;
            obj.pred_features.global3d = CONFIG.agent_pose_features_global3d;
            obj.pred_features.global3d_cum = CONFIG.agent_pose_features_global3d_cum;
            obj.nbr_stacked_cum_preds = CONFIG.agent_nbr_stacked_cum_preds;
            obj.use_canvas = CONFIG.agent_use_angle_canvas;
            obj.nbr_bins_azim = CONFIG.agent_nbr_bins_azim;
            obj.nbr_bins_elev = CONFIG.agent_nbr_bins_elev;
            obj.use_rig_canvas = CONFIG.agent_use_rig_canvas;
            obj.disable_temporal_fusion = CONFIG.always_reset_reconstruction;
            obj.pred_size = max(1, (obj.pred_features.camera2d * 15 * 2 + ...
                                    obj.pred_features.global3d * 15 * 3 + ...
                                    obj.pred_features.global3d_cum + 1));
            if obj.pred_features.global3d_cum
                obj.pred_size = obj.pred_size + obj.nbr_stacked_cum_preds * 15 * 3;
            end
            obj.compute_all_fix = CONFIG.evaluation_compute_all_fixations;
        end

        function reset_episode(obj)
            obj.next_frame();
            obj.nbr_actions_taken_seq = 0;
            obj.predictions_seq = {};
            obj.predictions_seq_ok = {};
            obj.predictions_seq_all = {};
            obj.predictions_seq_ok_all = {};
            obj.nbr_dets_all = [];
        end

        function next_frame(obj)
            if obj.disable_temporal_fusion
                obj.predictions_seq = {};
                obj.predictions_seq_ok = {};
                obj.predictions_seq_all = {};
                obj.predictions_seq_ok_all = {};
            end
            obj.predictions = {};
            obj.visited_cams = [];
            obj.pose_idxs = [];
            obj.reset_canvas();
            obj.nbr_actions_taken = 0;
            obj.curr_azim_bin = nan;
        end

        function reset(obj)
            obj.reset_episode();
            obj.reset_history();
        end

        function reset_history(obj)
            % Resets batch history (to be called after batch gradient updates)
            obj.history = struct;
            for i = 1 : numel(obj.data_names)
                obj.history.(obj.data_names{i}) = cell(obj.history_size, 1);
            end
        end
        
        function reset_canvas(obj)
            
            % Reset things related to angle canvas history
            if ~obj.use_canvas && ~obj.use_rig_canvas
                obj.canvas = 0;
                obj.rig_canvas = 0;
                return;
            end

            % Create azim and elev centers for assignments

            % azim
            tmp = linspace(-pi + pi / obj.nbr_bins_azim, ...
                           pi + pi / obj.nbr_bins_azim, ...
                           obj.nbr_bins_azim + 1);
            tmp = tmp(1 : end - 1);
            obj.azim_centers = [cos(tmp); sin(tmp)];

            % elev
            tmp = linspace(-obj.elev_angle_max + obj.elev_angle_max / obj.nbr_bins_elev, ...
                           obj.elev_angle_max + obj.elev_angle_max / obj.nbr_bins_elev, ...
                           obj.nbr_bins_elev + 1);
            obj.elev_centers = tmp(1 : end - 1);

            if obj.use_canvas
                % Init angle canvas
                obj.canvas = zeros(obj.nbr_bins_elev, obj.nbr_bins_azim);
                obj.prev_azim = 0;
            else
                obj.canvas = 0;
            end

            if obj.use_rig_canvas
                % Init rig canvas
                obj.rig_canvas = zeros(obj.nbr_bins_elev, obj.nbr_bins_azim);
            else
                obj.rig_canvas = 0;
            end
        end
        
        function set_elev_angle_max(obj, elev_angle_max)
            obj.elev_angle_max = elev_angle_max;
            obj.reset_canvas();
        end
            
        function update_hist(obj, episode, data_struct, action)
            % Convenience method for special call of the update_history
            % function (see below function)
            data_struct.action = [action.azim_angle; action.elev_angle; ...
                                  action.binary];
            data_struct.m = [obj.m_azim; obj.m_elev];
            obj.update_history(episode, data_struct);
        end

        function update_history(obj, episode, data_struct, do_stack)
            if ~exist('do_stack', 'var')
                do_stack = 0;
            end
            hist_idx = rem(episode - 1, obj.history_size) + 1;
            data_fields = fieldnames(data_struct);
            for i = 1 : numel(data_fields)
                data = data_struct.(data_fields{i});
                ndims_data = max(1, nnz(size(data) ~= 1));
                obj.data_ndims.(data_fields{i}) = ndims_data;
                nbr_dims = ndims(data) + ~nnz(size(data) == 1);
                if do_stack
                    obj.history.(data_fields{i}){hist_idx} = [obj.history.(data_fields{i}){hist_idx}; data];
                else
                    obj.history.(data_fields{i}){hist_idx} = ...
                        cat(nbr_dims, obj.history.(data_fields{i}){hist_idx}, data);
                end
            end
        end
        
        function update_hist_reward(obj, episode, rewards)
            data_struct.rewards_mises = rewards.mises;
            data_struct.rewards_binary = rewards.binary;
            obj.update_history(episode, data_struct, true);
        end
        
        function data = get_data_from_history(obj, data_name, do_cat)
            data = obj.history.(data_name);
            if obj.history_size > 1 && do_cat
                data = cat(obj.data_ndims.(data_name) + 1, data{:});
            end
        end
        
        function fill_data(obj, data_struct)
            data_fields = fieldnames(data_struct);
            for i = 1 : numel(data_fields)
                data = data_struct.(data_fields{i});
                obj.net.blobs(data_fields{i}).set_data(data);
            end 
        end
        
        function batch_data = get_batch_data(~, data, indices)
            otherdims = repmat({':'}, 1, ndims(data) - 1);
            batch_data = data(otherdims{:}, indices);
        end
        
        function data_ext = get_extended_data(obj, data, expand_dim)
            % expand_dim = 1 --> embeds the feature blob in one extra batch
            % dimension, otherwise expands along the already existing batch dimension
            if ~exist('expand_dim', 'var')
                expand_dim = 1;
            end
            if obj.batch_size > 1
                % Fill the rest with zero
                cum_data = data;
                size_cum_data = size(cum_data);
				if numel(size_cum_data) == ndims(data) && expand_dim
					if ~nnz(size(data) == 1)
						data_ext = zeros([size_cum_data, obj.batch_size]);
						otherdims = repmat({':'}, 1, ndims(data_ext) - 1);
						data_ext(otherdims{:}, 1) = cum_data;
					else
						tmp = size_cum_data(size(data) ~= 1);
						if isempty(tmp)
							tmp = 1;
						end
						data_ext = zeros([tmp, obj.batch_size]);
						otherdims = repmat({':'}, 1, ndims(data_ext) - 1);
						data_ext(otherdims{:}, 1 : size_cum_data(end)) = cum_data;
					end
				else
					data_ext = zeros([size_cum_data(1 : end - 1), obj.batch_size]);
					otherdims = repmat({':'}, 1, ndims(data_ext) - 1);
					data_ext(otherdims{:}, 1 : size_cum_data(end)) = cum_data;
				end
            else
                data_ext = data;
            end
        end
        
        function pred_camera = get_pred_camera_normalized(obj, pred_camera)
            pred_camera = bsxfun(@minus, pred_camera, obj.img_dims / 2); % center normalize
            pred_camera = bsxfun(@rdivide, pred_camera, obj.img_dims); % size normalize
        end
        
        function [action, data_in] = take_action(obj, states, greedy, env, person_idxs)
        
            global CONFIG
                                        
            nbr_persons = numel(person_idxs);
            if isempty(obj.predictions)
				obj.predictions = cell(1, nbr_persons);
                for i = 1 : nbr_persons
					obj.predictions{i} = [];
                end
            end
            
            nbr_steps = numel(obj.visited_cams);
            
            for i = 1 : nbr_persons
            
                % Go to current person
                person_idx = person_idxs(i);
                env.goto_person(person_idx);
                state = states{i};
                
                % "pred" is the current state's local 3D and 2D pose
                % predictions (with some normalization) etc
                pred = nan(obj.pred_size, 1);
                pred_ctr = 1;
                pred(pred_ctr) = state.pose_idx == -1;
                pred_ctr = pred_ctr + 1;
                if obj.pred_features.global3d > 0
                    pred_ctr_next = pred_ctr + numel(state.pred_state(:));
                    pred(pred_ctr : pred_ctr_next - 1) = 0.01 * state.pred_state(:);
                    pred_ctr = pred_ctr_next;
                end
                if obj.pred_features.camera2d > 0
                    pred_camera = obj.get_pred_camera_normalized(state.pred_camera_state);
                    pred_ctr_next = pred_ctr + numel(pred_camera(:));
                    pred(pred_ctr : pred_ctr_next - 1) = pred_camera(:);
                    pred_ctr = pred_ctr_next;
                end
                if obj.pred_features.global3d_cum > 0

                    % Cumulative pose estimate as feature
                    preds_with_curr = cat(3, obj.predictions{1}, state.pred);
                    [~, unique_idxs] = unique(obj.visited_cams);
                    preds_with_curr = preds_with_curr(:, :, unique_idxs);
                    cum_recon = nanmedian(preds_with_curr, 3);
                    pred_ctr_next = pred_ctr + numel(cum_recon(:));
                    pred(pred_ctr : pred_ctr_next - 1) = 0.01 * cum_recon(:);
                    pred_ctr = pred_ctr_next;
                    
                    % Append stacked history of cumulative pose estimates
                    if obj.nbr_stacked_cum_preds > 0
                        cum_preds = zeros(size(cum_recon, 1), size(cum_recon, 2), ...
                                          obj.nbr_stacked_cum_preds);
                        if isempty(obj.predictions_seq)
                            preds_seq = [];
                            nbr_prev_cum_preds = 0;
                        else
                            preds_seq = obj.predictions_seq{1};
                            nbr_prev_cum_preds = size(preds_seq, 3);
                        end
                        nbr_prev_cum_preds_to_use = min(nbr_prev_cum_preds, ...
                                                        obj.nbr_stacked_cum_preds);
                        cum_preds(:, :, 1 : nbr_prev_cum_preds_to_use) = ...
                            preds_seq(:, :, 1 : nbr_prev_cum_preds_to_use);
                        pred_ctr_next = pred_ctr + numel(cum_preds(:));
                        pred(pred_ctr : pred_ctr_next - 1) = cum_preds(:);
                    end
                end
                
                % Sanity check pose prediction state
                if any(isnan(pred))
                    error('check pred state!');
                end
                    
                % Store all predictions 
                obj.predictions{i} = cat(3, obj.predictions{i}, state.pred);
            end
                
            % Get auxiliary info
            aux = obj.nbr_actions_taken / obj.max_traj_len;
            is_new_frame = obj.nbr_actions_taken == 0;
            is_init_init_frame = obj.nbr_actions_taken_seq == 0;
            aux = [aux; is_new_frame; is_init_init_frame];
            nbr_dets = env.scene().get_nbr_detections(env.frame_idx, env.camera_idx);
            obj.nbr_dets_all(end + 1) = nbr_dets;
            aux = [aux; mean(obj.nbr_dets_all)];
                
            % Produce action probabilities
            blob = states{1}.blob;
            blob_ext = obj.get_extended_data(blob);
            pred_ext = obj.get_extended_data(pred);
            canvas_in = obj.canvas(:);
            canvas_ext = obj.get_extended_data(canvas_in);
            rig_in = obj.rig_canvas(:);
            rig_ext = obj.get_extended_data(rig_in);
            aux_ext = obj.get_extended_data(aux);
            elev_mult_ext = obj.get_extended_data(obj.elev_angle_max);
            data_struct = struct('data', blob_ext,  'pred', pred_ext, ...
                                 'canvas', canvas_ext, 'rig', rig_ext, ...
                                 'aux', aux_ext, 'elev_mult', elev_mult_ext);

            % Forward pass in policy network
            obj.fill_data(data_struct);
            obj.net.forward_prefilled();
            
            % Extract action probabilities and sample an action
            extra_args.nbr_actions = obj.nbr_actions;
            extra_args.env = env;
            action = obj.sample_action(greedy, extra_args);

            % Update number of actions taken
			obj.nbr_actions_taken = obj.nbr_actions_taken + 1;
			obj.nbr_actions_taken_seq = obj.nbr_actions_taken_seq + 1;

			% Below used for reward normalization purposes
            if ~greedy && is_new_frame
				obj.frame_starts = [obj.frame_starts, obj.nbr_actions_taken_seq];
            end

            % Update the visited angles canvas (for next time-step)
            obj.update_angle_canvas(action.azim_angle, action.elev_angle);
            obj.shift_canvas_ego(); % ego-centric relative to agent
            
            if action.binary || obj.nbr_actions_taken == obj.max_traj_len
                % At termination of current frame in sequence
                for i = 1 : nbr_persons
                    [pose_reconstruction, current_is_garbage] = ...
                        obj.helpers.get_recon_with_prev(obj.predictions, obj.pose_idxs, i, ...
                                                        obj.visited_cams, nbr_persons, ...
                                                        numel(obj.visited_cams), obj.predictions_seq, ...
                                                        obj.predictions_seq_ok, obj.predictions_seq_all, ...
                                                        obj.predictions_seq_ok_all, 1, 0, 1);
                    % Attach to container of all pose predictions over sequence
                    % We grow this in the "flipped" direction as below, for
                    % easier indexing
                    if numel(obj.predictions_seq_ok) == nbr_persons
						preds_seq_ok = obj.predictions_seq_ok{i};
						preds_seq = obj.predictions_seq{i};
                    else
						preds_seq_ok = [];
						preds_seq = [];
                    end
                    obj.predictions_seq{i} = cat(3, pose_reconstruction, preds_seq);
                    obj.predictions_seq_ok{i} = [1, preds_seq_ok];
                end
            end

            if obj.compute_all_fix
				obj.predictions_seq_all{end + 1} = cell(1, nbr_persons);
				obj.predictions_seq_ok_all{end + 1} = cell(1, nbr_persons);
                for i = 1 : nbr_persons
                    [pose_reconstruction, current_is_garbage] = ...
                        obj.helpers.get_recon_with_prev(obj.predictions, obj.pose_idxs, i, ...
                                                        obj.visited_cams, nbr_persons, ...
                                                        numel(obj.visited_cams), obj.predictions_seq, ...
                                                        obj.predictions_seq_ok, obj.predictions_seq_all, ...
                                                        obj.predictions_seq_ok_all, 1, 1);
					obj.predictions_seq_all{nbr_steps}{i} = cat(3, pose_reconstruction, obj.predictions_seq_all{nbr_steps}{i});
					obj.predictions_seq_ok_all{nbr_steps}{i} = [1, obj.predictions_seq_ok_all{nbr_steps}{i}];
                end
            end

            % Pack all data used into cell (used for backprop)
            data_in = struct('data', blob, 'pred', pred, 'canvas', ...
                             canvas_in, 'rig', rig_in, 'aux', aux, ...
                             'elev_mult', obj.elev_angle_max);
        end

        function action = sample_action(obj, greedy, extra_args)
            
            % Default args
            if ~exist('extra_args', 'var')
				extra_args = struct;
            end

            % Extract the 3 binary (sigmoid) probabilities
            output_batch_idx = 1;
            binary_probs = obj.net.blobs('sigmoid_binary').get_data();
            done_prob = binary_probs(1, output_batch_idx);

            % Sample binary (skip action for the next view for video agent)                        
            if isfield(extra_args, 'nbr_actions') && extra_args.nbr_actions > 0
                action.binary = (obj.nbr_actions_taken == extra_args.nbr_actions);
            else
				binary_prob = done_prob;
				if ~greedy
					action.binary = (rand() <= binary_prob);
				else
				    action.binary = (binary_prob >= 0.5);
				end
            end

            % Sample next viewing angles
            angles = obj.net.blobs('angles').get_data();
            angles = angles(:, output_batch_idx);
            azim_angle = angles(1);
            elev_angle = angles(2);
            if ~greedy
                azim_angle = obj.von_mises_dist.circ_randvm(azim_angle, obj.m_azim);
                elev_angle = obj.von_mises_dist.circ_randvm(elev_angle, obj.m_elev, 1, ...
                                                                [obj.elev_angle_max, ...
                                                                 -obj.elev_angle_max]);               
            end
            action.azim_angle = azim_angle;
            action.elev_angle = elev_angle;
        end
        
        function train_agent(obj, stats, ep)
            
            obj.last_trained_ep = ep;
            
            % Transform into appropriate tensors, i.e. from "grouped by
            % episode" to one big tensor
            S_blob = obj.get_data_from_history('data', true);
            S_pred = obj.get_data_from_history('pred', true);
            S_canvas = obj.get_data_from_history('canvas', true);
            S_rig = obj.get_data_from_history('rig', true);
            S_aux = obj.get_data_from_history('aux', true);
            S_elev_mult = obj.get_data_from_history('elev_mult', true);
            Actions = obj.get_data_from_history('action', false);
            Rewards_mises = obj.get_data_from_history('rewards_mises', false);
            Rewards_binary = obj.get_data_from_history('rewards_binary', false);
            Ms = obj.get_data_from_history('m', false);

            % Calculate number of total steps taken
            nbr_steps = size(S_blob, 4);

            % Initialize G-matrix
            G_mises = nan(nbr_steps, 1);
            G_binary = nan(nbr_steps, 1);
            
            % Keeps tracks which step in the episode each reward belongs to
            g_step_index = zeros(nbr_steps, 1);

            % Action order: [azim, elev, binary, mode]
            A = nan(3, nbr_steps);
            MS = nan(2, nbr_steps);

            % Create the G-matrices (used in loss in caffe,
            % implements the policy gradient theorem)
            step_counter = 1;
            seq_ctr = 0;
            for t = 1 : obj.history_size
                actions = Actions{t};
                rewards_mises = Rewards_mises{t};
                rewards_binary = Rewards_binary{t};
                ms = Ms{t};
                
                action_counter_tot = 1;
                for i = 1 : obj.sequence_length
                
                    % Extract actions for current active-view
                    idx = i + seq_ctr * obj.sequence_length;
                    if idx < numel(obj.frame_starts) && obj.frame_starts(idx + 1) > obj.frame_starts(idx)
                        curr_idxs = obj.frame_starts(idx) : obj.frame_starts(idx + 1) - 1;
                    else
                        curr_idxs = obj.frame_starts(idx) : numel(rewards_mises);
                    end
                    curr_actions = actions(:, curr_idxs);
                    
                    nbr_act = size(curr_actions, 2);
                    for action_counter = 1 : nbr_act
                        G_mises(step_counter) = rewards_mises(action_counter_tot);
                        G_binary(step_counter) = rewards_binary(action_counter_tot);
                        A(:, step_counter) = actions(:, action_counter_tot);
                        MS(:, step_counter) = ms(:, action_counter_tot);
                        g_step_index(step_counter) = action_counter;
                        step_counter = step_counter + 1;
                        action_counter_tot = action_counter_tot + 1;
                    end
                end
                seq_ctr = seq_ctr + 1;
            end

            % Perform mean-std reward normalization
            G_mises = obj.normalize_rewards_per_step(G_mises, g_step_index);
            G_binary = obj.normalize_rewards_per_step(G_binary, g_step_index);
            
            % Flip sign of G matrices, since we are minimizing a loss
            G_mises = -G_mises;
            G_binary = -G_binary; % proto implements C-E without minus!
            
            % Define random indices, to run over batch in random order, as
            % opposed to in the given sequential order, for more IID'ness
            % Currently we throw away the "remainder"
            rem_batch = rem(nbr_steps, obj.batch_size);
            all_indices = randperm(nbr_steps);
            
            % Train over batches
            for i = 1 : obj.batch_size : (nbr_steps - rem_batch)

                % Extract sub-batch for various data inputs
                indices = all_indices(i : i + obj.batch_size - 1);
                blob = obj.get_batch_data(S_blob, indices);
                pred = obj.get_batch_data(S_pred, indices);
                canvas_in = obj.get_batch_data(S_canvas, indices);
                rig = obj.get_batch_data(S_rig, indices);
                aux = obj.get_batch_data(S_aux, indices);
                elev_mult = obj.get_batch_data(S_elev_mult, indices);
				ms = MS(:, indices);
                               
                % Angle prediction reward
                reward_azim = G_mises(indices)';
                reward_elev = G_mises(indices)';
                reward_mises = [reward_azim; reward_elev];
				
                % Get 'fake labels' for mises rewards appropriately
                neg_angle_pred = -A(1 : 2, indices);
                               
                % Extract reward for binary action
                reward_binary = G_binary(indices)';
                binary_fake_label = A(3, indices);

                % Forward sub-batch. Note that below cell must be in the
                % same order as arguments were registered!
                data_struct = struct('data', blob, ...
                                     'pred', pred, ...
                                     'canvas', canvas_in, ...
                                     'rig', rig, ...
                                     'aux', aux, ...
                                     'elev_mult', elev_mult, ...
                                     'm', ms, ...
                                     'reward_binary', reward_binary, ...
                                     'reward_mises', reward_mises, ...
                                     'binary_fake_label', binary_fake_label, ...
                                     'binary_fake_label_neg', 1 - binary_fake_label, ...
                                     'neg_angle_pred', neg_angle_pred);
                obj.fill_data(data_struct);
                
                % Perform network update step
                obj.solver.step(1);  
                                  
                % Extract loss value
                batch_loss_mises = obj.net.blobs('loss_von_mises').get_data();
                batch_loss_binary = obj.net.blobs('loss_binary').get_data();
                stats.s('Loss').collect(batch_loss_mises + ...
                                        batch_loss_binary);
                stats.next_batch();
            end

            % Clear the history
            obj.reset_history();
            obj.frame_starts = [];
        end         
                        
		function update_angle_canvas(obj, azim_angle, elev_angle)
            if ~obj.use_canvas
				return;
            end
            new_azim = angle(exp(1i * (obj.prev_azim + azim_angle)));
            obj.prev_azim = new_azim;
            [~, azim_bin] = min(sum(bsxfun(@minus, obj.azim_centers, ...
                                           [cos(new_azim); sin(new_azim)]).^2));
            [~, elev_bin] = min(abs(obj.elev_centers - elev_angle));
            obj.canvas(elev_bin, azim_bin) = obj.canvas(elev_bin, azim_bin) + 1;
            obj.curr_azim_bin = azim_bin;
        end
  
        function init_rig_canvas(obj, env)
            if ~obj.use_rig_canvas
				return;
            end

            % Need to initialize camera rig based on the agent initial
            % global azimuth and elevation angles (known from env)
            global_angles_cam_init = env.global_angles_cam();
            azim_init = global_angles_cam_init(1);

            for camera_idx = 1 : env.scene().nbr_cameras
                global_angles_cam = env.scene().global_angles_cam(camera_idx);
                azim = angle(exp(1i * (global_angles_cam(1) - azim_init)));
                elev = global_angles_cam(2) - env.scene().camera_rig_individual.elev_angle_mean;
                [~, azim_bin] = min(sum(bsxfun(@minus, obj.azim_centers, ...
                                               [cos(azim); sin(azim)]).^2));
                [~, elev_bin] = min(abs(obj.elev_centers - elev));
                obj.rig_canvas(elev_bin, azim_bin) = obj.rig_canvas(elev_bin, azim_bin) + 1;
            end
            obj.rig_canvas = obj.rig_canvas / max(obj.rig_canvas(:));
        end

        function shift_canvas_ego(obj)
            if ~obj.use_canvas || ~obj.use_rig_canvas
                return;
            end
            azim_bin_diff = obj.curr_azim_bin - round((obj.nbr_bins_azim + 1) / 2);
            obj.canvas = circshift(obj.canvas, [0, -azim_bin_diff]);
            obj.rig_canvas = circshift(obj.rig_canvas, [0, -azim_bin_diff]);
            obj.prev_azim = obj.prev_azim + azim_bin_diff * (2 * pi / obj.nbr_bins_azim);
        end
        
        function [recon3Ds_nonan, recon3Ds] = get_reconstruction(obj, use_cum)

            % Args:
            % 
            % use_cum -- use previous temporal estimate into the recon?
            
			% Default args
            if ~exist('use_cum', 'var')
                use_cum = 0;
            end

            nbr_persons = size(obj.pose_idxs, 2);
            recon3Ds_nonan = cell(1, nbr_persons);
            recon3Ds = cell(1, nbr_persons);
            
            for i = 1 : nbr_persons
                % Fuse previous and current estimates
				[recon3D, ~] = obj.helpers.get_recon_with_prev(obj.predictions, obj.pose_idxs, i, ...
                                                               obj.visited_cams, nbr_persons, ...
                                                               numel(obj.visited_cams), obj.predictions_seq, ...
                                                               obj.predictions_seq_ok, obj.predictions_seq_all, ...
                                                               obj.predictions_seq_ok_all, use_cum, 0);
                recon3D_nonan = obj.helpers.infer_missing_joints(recon3D);
                recon3Ds_nonan{i} = recon3D_nonan;
                recon3Ds{i} = recon3D;
            end
        end
        
        function G = normalize_rewards(~, G)
            
            % Compute current mean, std
            g_nzi = ~isnan(G);
            g_nz = G(g_nzi);
            g_nz_mean = mean(g_nz);
            g_nz_std = std(g_nz);
            
            if g_nz_std > 1e-8
                G(g_nzi) = (g_nz - g_nz_mean) / g_nz_std;
            else
                G(g_nzi) = g_nz - g_nz_mean;
            end
            G(isnan(G)) = 0;
        end

        function G = normalize_rewards_per_step(obj, G, indices)
            for i = 1 : max(indices)
                inds_i = indices == i;
                G(inds_i, :) = obj.normalize_rewards(G(inds_i, :));
            end
        end
    end
end
