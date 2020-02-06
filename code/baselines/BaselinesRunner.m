classdef BaselinesRunner < handle
    % Baselines to compare Pose-DRL to
    
    properties
        baselines
        helpers
    end
    
    methods
        function obj = BaselinesRunner(baselines)

            obj.baselines = cell(1, numel(baselines));
            check_bl = strcmp(baselines, 'random');
            if any(check_bl)
                obj.baselines{check_bl} = Random(obj);
            end
            check_bl = strcmp(baselines, 'max_azim');
            if any(check_bl)
                obj.baselines{check_bl} = MaxAzim(obj);
            end
            check_bl = strcmp(baselines, 'oracle');
            if any(check_bl)
                obj.baselines{check_bl} = Oracle(obj);
            end
            obj.helpers = Helpers();
        end
        
        function errors = run(obj, env, out_sequence_agent, person_idxs)
            global CONFIG
            errors = cell(numel(obj.baselines), numel(out_sequence_agent));
            frame_idx_init = env.frame_idx;
            camera_idx_init = out_sequence_agent{1}.init_cam_idx;
            trackers = cell(1, numel(obj.baselines));
            for i = 1 : numel(obj.baselines)
                tracker = Tracker(env.scene(), person_idxs);
                tracker.start(frame_idx_init, camera_idx_init);
                trackers{i} = tracker;
            end
            for i = 1 : numel(obj.baselines)
                env.goto_frame(frame_idx_init);
                obj.baselines{i}.reset_prev_cums(tracker);
                out_cam = nan;
                for j = 1 : numel(out_sequence_agent)
                    env.goto_frame(frame_idx_init + CONFIG.sequence_step * (j - 1));
                    nbr_steps = max(1, out_sequence_agent{j}.traj_len);
                    if j == 1
                        % Only pick agent's init-init random camera for
                        % fairness in evaluation
                        env.goto_cam(camera_idx_init);
                    else
                        % In rest of sequence, give random start camera,
                        % since we don't want to bootstrap agent
                        % intelligence
                        if isnan(out_cam)
                            env.goto_cam(randi(env.scene().nbr_cameras))
                        else
                            env.goto_cam(out_cam)
                        end
                        trackers{i}.next(env.frame_idx, env.camera_idx);
                    end
                    name = obj.baselines{i}.name;
                    [error, preds, cams_visited, out_cam, errors_all] = ...
                        obj.baselines{i}.run(env, trackers{i}, nbr_steps); 
                    errors{i, j} = struct('name', name, ...
                                          'error', error, ...
                                          'predictions', preds, ...
                                          'visited_cams', cams_visited, ...
                                          'errors_all', errors_all);
                end
            end
            env.goto_frame(frame_idx_init);
            env.goto_cam(out_sequence_agent{1}.init_cam_idx);
        end
    end
end
