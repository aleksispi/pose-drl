classdef Random < Baseline
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        
        function obj = Random(baselines)
        	obj = obj@Baseline(baselines, 'random');
        end
        
        function [recon_err, preds, cams_visited, out_camera, recon_errs_all] = run(obj, env, tracker, nbr_cameras)
            
            nbr_persons = tracker.nbr_persons;
            preds_all = nan(15, 3, obj.max_traj_len, nbr_persons);
            pose_idxs_all = nan(obj.max_traj_len, nbr_persons);
            cams_visited = nan(1, obj.max_traj_len);
            i = 1;
            while i <= obj.max_traj_len
                if i > 1
                    env.goto_cam(next_cam_idx);
                    tracker.next(env.frame_idx, env.camera_idx);
                end
                cams_visited(i) = env.camera_idx;
                next_cam_idx = randi(env.scene().nbr_cameras);
                while i <= env.scene().nbr_cameras && any(cams_visited == next_cam_idx)
                    next_cam_idx = randi(env.scene().nbr_cameras);
                end
                pose_idxs = tracker.get_detections();
                pose_idxs_all(i, :) = pose_idxs;
                
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
                
                % Check for termination
                if i == nbr_cameras
                    break;
                end
                i = i + 1;
            end
            cam_counter = min(i, obj.max_traj_len);
            [recon_err, preds, cams_visited, out_camera] = ...
                obj.compute_cum_recon_err(preds_all, env, tracker, ...
                                          pose_idxs_all, cam_counter, cams_visited);
            if obj.compute_all_fixations
                recon_errs_all = zeros(1, cam_counter);
                for i = 1 : cam_counter
                    [recon_err_i, ~, ~, ~] = ...
                        obj.compute_cum_recon_err(preds_all, env, tracker, ...
                                                  pose_idxs_all, cam_counter, cams_visited, i);
                    recon_errs_all(i) = recon_err_i;
                end
            else
                recon_errs_all = zeros(1, obj.max_traj_len);
            end
        end
    end
end

