classdef MaxAzim < Baseline
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    methods
        
        function obj = MaxAzim(baselines)
        	obj = obj@Baseline(baselines, 'max_azim');
        end
        
        function [recon_err, preds, cams_visited, out_camera, recon_errs_all] = run(obj, env, tracker, nbr_cameras)
            
            % Specify max elev spread for this scene
            elev_angle_max = env.scene().camera_rig_individual.elev_angle_span_half;
            
            % Initialize containers
            nbr_persons = tracker.nbr_persons;
            preds_all = nan(15, 3, obj.max_traj_len, nbr_persons);
            pose_idxs_all = nan(obj.max_traj_len, nbr_persons);
            cams_visited = nan(1, obj.max_traj_len);
            
            % Get initial state and camera index
            pose_idxs = tracker.get_detections();
            pose_idxs_all(1, :) = pose_idxs;

            cam_idx = env.camera_idx;
            cams_visited(1) = env.camera_idx;
            for j = 1 : nbr_persons

                % Extract current pose index
                pose_idx = pose_idxs(j);

                % Goto current person
                env.goto_person(tracker.person_idxs(j));

                % Get 2D pose prediction and data blob
                [pred, ~] = env.get_current_predictor(pose_idx);

                state = env.get_state(pose_idx, nan, pred);
                preds_all(:, :, 1, j) = state.pred;
            end
            
            % Find the global azim angle of the given camera and set 
            % azim angle step length (unidistant max azim steps)
            angle_global = env.scene().global_angles_cam(cam_idx);
			azim_global = angle_global(1);
            azim_step = 2 * pi / nbr_cameras;
            
            for i = 2 : nbr_cameras
            
				% Update azim angle (unidistant steps)
				azim_global = azim_global + azim_step;
				azim_global = angle(cos(azim_global) + 1i * sin(azim_global));
            
				% Update elev angle (randomly)
				elev_local = elev_angle_max * 2 * (rand() - 0.5);
				[~, elev_global] = env.agent_angles_to_global(nan, elev_local);
            
				% Go to new (azim, elev) angles
				env.goto_cam_mises(azim_global, elev_global);
                
                % Ensure new camera location!
                while any(cams_visited == env.camera_idx)
                    try_cam_idx = randi(env.scene().nbr_cameras);
                    env.goto_cam(try_cam_idx);
                end
                
                tracker.next(env.frame_idx, env.camera_idx);
                pose_idxs = tracker.get_detections();
                pose_idxs_all(i, :) = pose_idxs;
                
                cams_visited(i) = env.camera_idx;
                for j = 1 : nbr_persons
                    
                    % Extract current pose index
                    pose_idx = pose_idxs(j);

                    % Goto current person
                    env.goto_person(tracker.person_idxs(j));
                    
                    % Get 2D pose prediction and data blob
                    [pred, ~] = env.get_current_predictor(pose_idx);

                    state = env.get_state(pose_idx, nan, pred);
                    preds_all(:, :, i, j) = state.pred;
                end
            end
            [recon_err, preds, cams_visited, out_camera] = ...
                obj.compute_cum_recon_err(preds_all, env, tracker, ...
                                          pose_idxs_all, nbr_cameras, cams_visited);
            if obj.compute_all_fixations
                recon_errs_all = zeros(1, nbr_cameras);
                for i = 1 : nbr_cameras
                    [recon_err_i, ~, ~, ~] = ...
                        obj.compute_cum_recon_err(preds_all, env, tracker, ...
                                                  pose_idxs_all, nbr_cameras, cams_visited, i);
                    recon_errs_all(i) = recon_err_i;
                end
            else
                recon_errs_all = zeros(1, obj.max_traj_len);
            end
        end
    end
end

