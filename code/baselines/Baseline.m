classdef Baseline < handle
    
    properties
        name
        id
        baselines
        helper
        max_traj_len
        always_reset
        prev_cum_preds
        prev_cum_preds_all
        compute_all_fixations
    end
    
    methods
        
        function obj = Baseline(baselines, name)
            global CONFIG
            obj.baselines = baselines;
            obj.name = name;
            obj.helper = Helpers();
            obj.max_traj_len = CONFIG.agent_max_traj_len;
            obj.always_reset = CONFIG.always_reset_reconstruction;
            obj.compute_all_fixations = CONFIG.evaluation_compute_all_fixations;
        end
        
        function [recon_err, preds, cams_visited, prev_cum_pred] = ...
                    run(obj, env, nbr_steps, prev_cum_pred) %#ok<*INUSL,*INUSD>
            recon_err = nan;
            preds = nan;
            cams_visited = nan;
        end
        
        function reset_prev_cums(obj, tracker)
            obj.prev_cum_preds = cell(obj.max_traj_len, tracker.nbr_persons);
            obj.prev_cum_preds_all = cell(obj.max_traj_len, tracker.nbr_persons);
            for traj = 1 : obj.max_traj_len
                for pid = 1 : tracker.nbr_persons
                    obj.prev_cum_preds{traj, pid} = nan;
                    obj.prev_cum_preds_all{traj, pid} = nan;
                end
            end
        end
        
        function [recon_err, preds, cams_visited, out_camera] = ...
                compute_cum_recon_err(obj, preds_all, env, tracker, ...
                                      pose_idxs_all, cam_counter, cams_visited, ...
                                      nbr_steps, update_prev_cum)
            
            % Default args
            if ~exist('nbr_steps', 'var')
                nbr_steps = cam_counter;
            end
            if ~exist('update_prev_cum', 'var')
                update_prev_cum = 1;
            end
            
            nbr_persons = tracker.nbr_persons;
            cams_visited = cams_visited(1 : cam_counter);
            recon_errs = nan(1, nbr_persons);
            for i = 1 : nbr_persons
                person_idx = tracker.person_idxs(i);
                env.goto_person(person_idx);
                prev_cum_pred = obj.prev_cum_preds{nbr_steps, i};
                preds = preds_all(:, :, 1 : nbr_steps, i);
                pose_idxs = pose_idxs_all(1 : nbr_steps, i);
                qualifier = (pose_idxs ~= -1);
                current_is_garbage = ~isempty(qualifier) && all(qualifier == 0);
                if current_is_garbage
                    % Ensure non-collapse
                    qualifier(1) = 1;
                end
                
                if ~isscalar(prev_cum_pred)
                    prev_cum_pred = obj.helper.trans_hip_coord(prev_cum_pred, preds(:, :, 1));
                    preds = cat(3, preds, prev_cum_pred);
                end
                if ~obj.always_reset && ~isscalar(prev_cum_pred) 
                    if current_is_garbage
                        % Only set current to previous cum-estimate, as the
                        % current is garbage --- qualfier 0s
                        qualifier = [0 * qualifier; 1];
                    else
                        % Ensure that the previous pose estimate is fused
                        % together with current
                        qualifier = [qualifier; 1]; %#ok<*AGROW>
                    end
                    preds(:, :, ~qualifier) = nan;
                    recon3D = nanmedian(preds, 3);
                elseif ~isscalar(prev_cum_pred)
                    recon3D = nanmedian(preds(:, :, 1 : end - 1), 3);
                else
                    recon3D = nanmedian(preds, 3);
                end
                recon3D_nonan = obj.helper.infer_missing_joints(recon3D);
                cum_recon_err = env.get_recon_error(recon3D_nonan);
                
                recon_errs(i) = cum_recon_err;
                if update_prev_cum
                    obj.prev_cum_preds{nbr_steps, i} = recon3D;
                end
            end
            preds = preds_all(:, :, 1 : nbr_steps, :);
            recon_err = mean(recon_errs);
            out_camera = cams_visited(end);
        end
    end
end

