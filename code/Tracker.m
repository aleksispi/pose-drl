classdef Tracker < handle
    % Tracks target persons
    
    properties
        person_idxs
        nbr_persons
        scene
        view_history
        match_cost_thresh
        inst_feats_source
    end
    
    methods
        
        function obj = Tracker(scene, person_idxs)
            global CONFIG
            obj.person_idxs = person_idxs;
            obj.nbr_persons = numel(person_idxs);
            obj.scene = scene;
            obj.view_history = {};
            obj.match_cost_thresh = CONFIG.panoptic_match_cost_thresh;
        end
        
        function start(obj, frame_idx, camera_idx)
            % Start tracking person in frame, camera.
            % Throws error if invalid start frame.
            global CONFIG
            
            
            % Pre-compute centroid features for the source image
            obj.inst_feats_source = nan(obj.scene.nbr_persons, 50);    
            for person_idx = 1 : size(obj.inst_feats_source, 1)
                models = obj.build_apperance_model(frame_idx, person_idx, ...
                                                   CONFIG.panoptic_nbr_appearance_samples);
                obj.inst_feats_source(person_idx, :) = models;
            end
            
            % Infer init-init detections
            pose_idxs = obj.infer_detection(frame_idx, camera_idx);
            obj.view_history{end + 1} = struct('frame', frame_idx, ...
                                               'camera', camera_idx, ...
                                               'pose_idxs', pose_idxs);
        end

        function pose_idxs = next(obj, frame, camera)
            pose_idxs = obj.infer_detection(frame, camera);
            obj.view_history{end + 1} = struct('frame', frame, ...
                                               'camera', camera, ...
                                               'pose_idxs', pose_idxs);     
        end

        function remove(obj)
            % Reverts last next operation
            obj.view_history = obj.view_history(1 : end - 1);
        end
        
        function pose_idxs = get_detections(obj)
            % Returns pose indices for current view
            pose_idxs = obj.view_history{end}.pose_idxs;
        end
        
        function bbox = get_detection_bbox(obj, pose_idx, view_idx)
            % Returns current detection bbox for current view (but can also
            % return it for any other detection in current view if
            % specifically passed as an argument)
            if ~exist('view_idx', 'var')
                view_idx = numel(obj.view_history); 
            end
            frame_idx = obj.view_history{view_idx}.frame;
            camera_idx = obj.view_history{view_idx}.camera;
            if ~exist('pose_idx', 'var')
                pose_idx = obj.view_history{view_idx}.pose_idx;
            end
            if pose_idx > 0
                bbox = obj.scene.pose_to_bbox(...
                        obj.scene.pose_cache_2d{frame_idx, ...
                                                camera_idx}{pose_idx});
            else
                bbox = zeros(1, 4);
            end
        end
        
        function bboxes = get_detection_bboxes(obj, pose_idxs, view_idx)
            % Returns all detection boxes for the current view (or any
            % other view, which may optionally be specified)
            if ~exist('view_idx', 'var')
                view_idx = numel(obj.view_history); 
            end
            frame_idx = obj.view_history{view_idx}.frame;
            camera_idx = obj.view_history{view_idx}.camera;
            nbr_detections = numel(obj.scene.pose_cache_2d{frame_idx, camera_idx});
            bboxes = nan(numel(pose_idxs), 4);
            if ~exist('pose_idxs', 'var')
                pose_idxs = 1 : nbr_detections;
            end
            for i = 1 : numel(pose_idxs)
                bboxes(i, :) = obj.get_detection_bbox(pose_idxs(i), view_idx);
            end
        end 
       
        function appearance_model = build_apperance_model(obj, frame_idx, ...
                                                          person_idx, n)
            % Finds n source features to use for tracking
            global CONFIG
            
            instance_feats = zeros(n, 50);
            nbr_found = 0;
            
            % Frame indices for appearance model backwards in time
            max_val = max(1, frame_idx - 1);
            min_val = max(1, frame_idx - 11 * CONFIG.sequence_step);
            allowed_frame_idxs_backward = min_val : CONFIG.sequence_step : max_val;
            
            % Frame indices for appearance model forwards in time
            min_val = min(obj.scene.nbr_frames, frame_idx + 11 * CONFIG.sequence_step);
            max_val = min(obj.scene.nbr_frames, frame_idx + 21 * CONFIG.sequence_step);
            allowed_frame_idxs_forward = min_val : CONFIG.sequence_step : max_val;
            
            % Train appearance model with most data (either before or after
            % the current active-sequence)
            if numel(allowed_frame_idxs_backward) >= numel(allowed_frame_idxs_forward)
                allowed_frame_idxs = allowed_frame_idxs_backward;
            else
                allowed_frame_idxs = allowed_frame_idxs_forward;
            end
            
            frame_idxs = repmat(allowed_frame_idxs, 1, ...
                            round(obj.scene.nbr_frames / numel(allowed_frame_idxs)));
            cam_idxs = 1 : obj.scene.nbr_cameras;
            for i = 1 : numel(frame_idxs)          
                if nbr_found == n
                    break
                end
                cam_idx = cam_idxs(rem(i - 1, obj.scene.nbr_cameras) + 1);
                [pose_idx, iou] = obj.scene.get_gt_detection(frame_idxs(i), ...
                                                             cam_idx, person_idx);
                
                if pose_idx == -1 || iou < CONFIG.panoptic_min_iou
                    continue
                end
                if obj.scene.is_gt_occluded(frame_idxs(i), cam_idx, person_idx)
                    continue
                end
                
                if ~isempty(obj.scene.instance_features{frame_idxs(i), cam_idx})
                    nbr_found = nbr_found + 1;
                    instance_feats(nbr_found, :) = ...
                        obj.scene.instance_features{frame_idxs(i), cam_idx}(pose_idx, :);
                end
            end
            appearance_model = median(instance_feats(1 : nbr_found, :), 1);
        end
        
        function pose_idxs = infer_detection(obj, target_frame, target_camera)
            % Inferred detection is always in relation to the initial
            % target detection, hence why we need the fully dense matches
            % Get features for source (from)
            cost_matrix = obj.compute_instance_costs(obj.inst_feats_source, ...
                                                     target_frame, target_camera);
            [pose_idxs, ~] = obj.hungarian_match(cost_matrix);
            pose_idxs = pose_idxs(obj.person_idxs);
        end
        
        function cost_matrix = compute_instance_costs(obj, inst_feats_from, ...
                                                      f_to, c_to)              
            % Instance matching

            % Get features for target (to)
            inst_feats_to = obj.scene.instance_features{f_to, c_to};
                
            % Construct instance cost matrix
            cost_matrix = pdist2(inst_feats_from, inst_feats_to);                        
        end

        function [match, cost] = hungarian_match(obj, cost_matrix)
            [match, ~] = munkres(cost_matrix);
            cost = nan(size(match));
            for m = 1 : numel(match)
                if match(m) > 0
                    cost(m) = cost_matrix(m, match(m));
                end
            end
            match(or(match < 1, cost > obj.match_cost_thresh)) = -1; 
        end    
    end
end