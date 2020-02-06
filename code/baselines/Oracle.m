classdef Oracle < Baseline
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        
        function obj = Oracle(baselines)
        	obj = obj@Baseline(baselines, 'oracle');
        end
        
        function [recon_err, preds, cams_visited, out_camera, recon_errs_all] = run(obj, env, tracker, nbr_cameras)
            
            % Initialize containers
            nbr_persons = tracker.nbr_persons;
            preds_all = nan(15, 3, obj.max_traj_len, nbr_persons);
            pose_idxs_all = nan(obj.max_traj_len, nbr_persons);
            cams_visited = nan(1, obj.max_traj_len);
            
            % Extract things relevant from the environment
            if tracker.nbr_persons == 1                
                best_cam_ord = env.scene().best_cam_ord(tracker.person_idxs, env.frame_idx, :);
            else
                best_cam_ord = env.scene().best_cam_ord(end, env.frame_idx, :);
            end
            
            % Run algorithm
            i = 1;
            while i <= obj.max_traj_len
                if i > 1
                    
                    % Greedily add camera which reduces avearge 3D error the most
                    min_err = 999999;
                    min_idx = 0;
                    best_cams_setdiff = setdiff(best_cam_ord, cams_visited, 'stable');
                    for j = 1 : numel(best_cams_setdiff)
                        
                        % Goto candidate camera
                        camera_idx = best_cams_setdiff(j);
                        env.goto_cam(camera_idx);
                        tracker.next(env.frame_idx, env.camera_idx);
                        pose_idxs = tracker.get_detections();
                        pose_idxs_candidate = pose_idxs_all;
                        pose_idxs_candidate(i, :) = pose_idxs;
                
                        preds_candidate = preds_all;
                        
                        for jj = 1 : nbr_persons

                            % Extract current pose index
                            pose_idx = pose_idxs(jj);

                            % Goto current person
                            env.goto_person(tracker.person_idxs(jj));

                            % Get 2D pose prediction and data blob
                            [pred, ~] = env.get_current_predictor(pose_idx);

                            state = env.get_state(pose_idx, nan, pred);
                            preds_candidate(:, :, i, jj) = state.pred;
                        end
                        
                        % Check if new best camera
                        [recon_err, ~, ~, ~] = ...
                            obj.compute_cum_recon_err(preds_candidate, env, tracker, ...
                                                      pose_idxs_candidate, i, cams_visited, i, 0);
                        
                        if recon_err < min_err
                            min_err = recon_err;
                            min_idx = camera_idx;
                        end
                        
                        % Revert
                        tracker.remove();
                    end
                    
                    % Goto winning camera
                    camera_idx = min_idx;
                    env.goto_cam(camera_idx);
                    tracker.next(env.frame_idx, env.camera_idx);
                else
                    camera_idx = env.camera_idx;
                end
                
                % Get 2D pose prediction and data blob
                pose_idxs = tracker.get_detections();
                pose_idxs_all(i, :) = pose_idxs;
                
                cams_visited(i) = camera_idx;
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
