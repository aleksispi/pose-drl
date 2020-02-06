classdef Scene < handle
    % Represents a scene in panoptic, with all frames and cameras,
    % including annotations

    properties
        camera_path
        scene_name
        camera_names
        nbr_cameras
        frame_names
        nbr_frames
        nbr_persons
        frame_annots
        body_id_2_idx
        camera_calibrations
        camera_annots_cache
        predictor_cache
        predictor_cache_loaded
        camera_rig
        % if at some point camera_rig refitted, keep original
        camera_rig_individual
        predictor

        helpers

        dataset_path
        dataset_cache

        % camera rank cache
        best_cam_ord

        % detection related
        detections
        match_cost_thresh
        pose_idx
        instance_features

        % 2d/3d pose cache
        pose_cache_2d % 2d cache
        pose_cache_3d
    end

    methods

        function obj = Scene(scene_name, dataset_path, dataset_cache, predictor)

            global CONFIG


            obj.predictor = predictor;
            obj.dataset_path = dataset_path;
            obj.dataset_cache = dataset_cache;

            % Constructor
            addpath('code/panoptic/json-matlab');

            % Get some helper functions
            obj.helpers = Helpers();

            % List all  and frames
            cam_path = strcat(obj.dataset_path, scene_name, '/', ...
                              CONFIG.dataset_video_format, 'Imgs/');

            [camera_names, ~] = obj.helpers.list_filenames(cam_path); %#ok<*PROP>
            frame_path = strcat(cam_path, camera_names{1});
            [frame_names, nbr_frames] = obj.helpers.list_filenames(frame_path);

            % Remove camera idx from frame names and keep only frame id.
            for i = 1 : nbr_frames
                name = strsplit(frame_names{i}, '_');
                frame_names{i} = name{3};
            end

            % Set some fields
            obj.scene_name = scene_name;
            obj.nbr_cameras = numel(camera_names);
            obj.camera_names = camera_names;
            obj.frame_names = frame_names;
            obj.nbr_frames = nbr_frames;
            obj.camera_path = cam_path;
            obj.predictor_cache_loaded = false;
            obj.match_cost_thresh = CONFIG.panoptic_match_cost_thresh;

            % Load cache
            obj.init_cached_properties();

            % Init predictor cache struct
            obj.predictor_cache = struct();
            for i = 1 : numel(CONFIG.predictor_collect_blobs_from)
               obj.predictor_cache.(CONFIG.predictor_collect_blobs_from{i}) = cell(obj.nbr_cameras, obj.nbr_frames);
            end

            % Setup camera rig settings
            obj.camera_rig = CameraRig(obj.camera_calibrations);

            % Fix camera calibs and annots to match camera names
            obj.camera_names = camera_names;
            obj.nbr_cameras = numel(obj.camera_names);
            obj.camera_calibrations = obj.camera_calibrations(1 : obj.nbr_cameras);
            obj.camera_annots_cache = obj.camera_annots_cache(:, 1 : obj.nbr_cameras, :);

            % Adjust camera calibs based on if any cameras are excluded
            obj.camera_rig.update_rig(obj.camera_calibrations)
            obj.camera_rig_individual = obj.camera_rig;
            fprintf('Scene elevation angle span (rad): %.15f\n', obj.camera_rig.elev_angle_span_half);
        end

        function check_data_consistency(obj)
            if isempty(obj.pose_cache_3d)
                return;
            end

            fprintf('Verifying scene data integrity...');

            % All frames and cameras should have instance features
            if size(obj.instance_features, 1) ~= obj.nbr_frames || ...
                size(obj.instance_features, 2) ~=  obj.nbr_cameras
                error('Not enough instance_features!!');
            end

            fprintf('ok!\n');
        end

        function set_predictor(obj, predictor)
            obj.predictor = predictor;
        end

        function [pred, blob] = get_predictor(obj, frame_idx, camera_idx, pose_idx, opts)
            % Runs some network (e.g. MubyNet) to extract feature for the
            % frame indexed by frame_idx, as seen from the camera indexed
            % by camera_idx
            global CONFIG
            if ~exist('opts', 'var')
               opts = struct('raw_pred_format', 0);
            end
            if opts.raw_pred_format == 1
                error('Not yet implemented');
            end

			% Get the deep feature cache for image
			blob = obj.predictor_cache.(CONFIG.predictor_use_blob){camera_idx, frame_idx};

			if pose_idx == -1
				pred = nan(15, 3);
			elseif ~isscalar(obj.pose_cache_3d)
				pred = obj.pose_cache_3d{frame_idx, camera_idx}{pose_idx};

				if size(pred, 1) == 17
					pred = 1000 * pred(CONFIG.panoptic_joint_mapping, :) ...
									* CONFIG.panoptic_scaling_factor;
				end
			else
				error('No 3d pose cache exists!');
			end
        end

        function [pred, feature_blobs] = run_predictor(obj, frame_idx, camera_idx, pose_idx)
            % Get current image and run through pose predictor
            img = obj.get_img(frame_idx, camera_idx);

            % Crop  image to get only human
            bbox = obj.get_detection_info(frame_idx, camera_idx, pose_idx).bbox;

            global CONFIG

            % Pad a bit extra.
            bbox([1,3]) = bbox([1,3]) - CONFIG.panoptic_crop_margin';
            bbox([2,4]) = bbox([2,4]) + CONFIG.panoptic_crop_margin';

            dims = CONFIG.dataset_img_dimensions;

            bbox([1, 3]) =  max(bbox([1, 3]), 1);
            bbox(2) =  min(bbox(2), dims(1));
            bbox(4) =  min(bbox(4), dims(2));

            img = obj.crop_human(img, bbox);
            [pose3D, feature_blobs] = obj.predictor.predict(img);

            % Return only prediction from final stage only
            pred = pose3D{end};
        end

        function best_idx = get_person_from_detection(obj, frame_idx, ...
                                                      camera_idx, ...
                                                      pose_idx)
            % Should find closest matching GT bbox to 2d predicted bbox

            if pose_idx < 1
                best_idx = -1;
                return;
            end

            pose_bbox = obj.pose_to_bbox(obj.pose_cache_2d{frame_idx, camera_idx}{pose_idx});
            pose_bbox = [pose_bbox(3), pose_bbox(1), pose_bbox(4), pose_bbox(2)];

            best_idx = -1;
            best_iou = 0;
            for person_idx = 1:obj.nbr_persons
                pose = obj.get_projected_annot(frame_idx, camera_idx, person_idx);
                annot_bbox = obj.pose_to_bbox(pose);
                annot_bbox = [annot_bbox(3), annot_bbox(1), annot_bbox(4), annot_bbox(2)];
                iou = obj.helpers.iou(pose_bbox, annot_bbox);

                if iou > best_iou
                    best_idx = person_idx;
                    best_iou = iou;
                end
            end
        end

        function [best_idx, best_iou] = get_gt_detection(obj, frame_idx, camera_idx, person_idx)
            % Retrives the best matching detection to a GT box. Inverse
            % of get_person_from_detection
            best_idx = -1;
            best_iou = 0;
			pose = obj.get_projected_annot(frame_idx, camera_idx, person_idx);
			annot_bbox = obj.pose_to_bbox(pose);
			annot_bbox = [annot_bbox(3), annot_bbox(1), annot_bbox(4), annot_bbox(2)];
			for p_idx = 1 : numel(obj.pose_cache_3d{frame_idx, camera_idx})
				bbox =  obj.pose_to_bbox(obj.pose_cache_2d{frame_idx, camera_idx}{p_idx});
				bbox = [bbox(3), bbox(1), bbox(4), bbox(2)];
				iou = obj.helpers.iou(bbox, annot_bbox);
				if iou > best_iou
					best_idx = p_idx;
					best_iou = iou;
				end
			end
        end

        function yn = is_gt_occluded(obj, frame_idx, camera_idx, person_idx)
            % Checks if the target's projected GT-box is occluded by other GT boxes
            global CONFIG
            pose = obj.get_projected_annot(frame_idx, camera_idx, person_idx);
            bbox = obj.pose_to_bbox(pose);
            bbox = [bbox(3), bbox(1), bbox(4), bbox(2)];
            for pid = 1:obj.nbr_persons
                if pid ~= person_idx
                    pose = obj.get_projected_annot(frame_idx, camera_idx, pid);
                    bbox2 = obj.pose_to_bbox(pose);
                    bbox2 = [bbox2(3), bbox2(1), bbox2(4), bbox2(2)];
                    iou = obj.helpers.iou(bbox, bbox2);

                    if iou > CONFIG.appearance_gt_iou_occluded
                        % At this stage, check if the target GT-box is
                        % behind (occluded) or in front (un-occluded)
                        pose3d_target = obj.get_camera_annot(frame_idx, camera_idx, person_idx);
                        pose3d_other = obj.get_camera_annot(frame_idx, camera_idx, pid);
                        if nnz(pose3d_target(:, 3) < 0) < nnz(pose3d_other(:, 3) < 0) || norm(pose3d_target(:, 3), 2) >= norm(pose3d_other(:, 3), 2)
                            yn = 1;
                            return;
                        end
                    end
                end
            end
            yn = 0;
        end

        function nbr_dets = get_nbr_detections(obj, frame_idx, camera_idx)
            % Returns number of detections for view indexed by frame_idx,
            % camera_idx
            nbr_dets = numel(obj.pose_cache_2d{frame_idx, camera_idx});
        end
        
        function path = img_path(obj, frame_idx, camera_idx)
            % Return image path to frame indexed by frame_idx, camera
            % indexed by camera_idx
            path = strcat(obj.camera_path, obj.camera_names{camera_idx}, ...
                '/', obj.camera_names{camera_idx}, '_', obj.frame_names{frame_idx});
        end

        function img = get_img(obj, frame_idx, camera_idx)
            % Returns RGB image associated with current view
            img = imread(obj.img_path(frame_idx, camera_idx));
        end

        function img_cropped = crop_human(~, img, bbox)
            % Returns the cropped version of the image
            dims = size(img);

            bbox = round(bbox);

            bbox([1, 3]) =  max(bbox([1, 3]), 1);
            bbox(2) =  min(bbox(2), dims(1));
            bbox(4) =  min(bbox(4), dims(2));

            img_cropped = img(bbox(1) : bbox(2), bbox(3) : bbox(4), :);

            dims_crop = size(img_cropped);
            if (dims_crop(1) <= 1 || dims_crop(2) <= 1)
                warning('scene.crop_human: Image too small (less than 1 pixel) -- keeping orig image size')
                img_cropped = img;
            end
        end

        function init_cached_properties(obj)
            % Caches (saves as mat-files) camera calibrations and
            % frame-level annotations

            % Create cache path if it doesn't exist
            if ~exist(obj.dataset_cache, 'dir')
                mkdir(obj.dataset_cache);
            end

            % Cache camera calibrations
            calib_path = strcat(obj.dataset_cache, '/', obj.scene_name, ...
                                '_calib.mat');
            if exist(calib_path, 'file') == 2
                S = load(calib_path);
                obj.camera_calibrations = S.camera_calibrations;
            else
                fprintf('Camera calibration cache not found -- creating!\n')
                camera_calibrations = obj.load_camera_calibrations();
                obj.camera_calibrations = camera_calibrations;
                save(calib_path, 'camera_calibrations');
            end

            % Cache frame-level annotations
            annot_path = strcat(obj.dataset_cache, '/', obj.scene_name, '_annot.mat');
            if exist(annot_path, 'file') == 2
                S = load(annot_path);
                obj.frame_annots = S.frame_info.frame_annots;
                obj.body_id_2_idx = S.frame_info.body_id_2_idx;
                obj.nbr_persons = size(obj.frame_annots, 1);
            else
                fprintf('Frame annot cache not found -- creating!\n')
                [frame_annots, body_id_2_idx] = obj.load_annotations();
                obj.frame_annots = frame_annots;
                obj.body_id_2_idx = body_id_2_idx;
                frame_info.frame_annots = frame_annots;
                frame_info.body_id_2_idx = body_id_2_idx; %#ok<*STRNU>
                save(annot_path, 'frame_info');
            end

            % Cache also camera-level annots
            obj.init_cached_camera_annots();
        end

        function [bbox, inside_frac] = pose_to_bbox(~, pose)
            global CONFIG

            if nnz(pose) == 0
                bbox = [1, 1, 1, 1];
                inside_frac = 0;
            else
                pose = round(pose);
                coord_min = min(pose);
                coord_max = max(pose);
                coord_min(1) = coord_min(1) - CONFIG.panoptic_crop_margin(1);
                coord_min(2) = coord_min(2) - CONFIG.panoptic_crop_margin(2);
                coord_max(1) = coord_max(1) + CONFIG.panoptic_crop_margin(1);
                coord_max(2) = coord_max(2) + CONFIG.panoptic_crop_margin(2);
                bbox = [coord_min(2), coord_max(2), coord_min(1), coord_max(1)];
                bbox = round(bbox);
                dims = CONFIG.dataset_img_dimensions;
                area_before_clip = (bbox(2) - bbox(1)) * (bbox(4) - bbox(3));
                bbox([1, 3]) =  max(bbox([1, 3]), 1);
                bbox(2) =  min(bbox(2), dims(1));
                bbox(4) =  min(bbox(4), dims(2));
                area_after_clip = (bbox(2) - bbox(1)) * (bbox(4) - bbox(3));
                inside_frac = area_after_clip / area_before_clip;
            end
        end

        function init_detections(obj)
            fprintf('Loading detections / matches / instance features...\n');

            global CONFIG

            % Detections and matchings (matchings contains detections)
            % frame, camera, person and detection ids are all 0-indexed!
            instance_path = strcat(obj.dataset_cache, obj.scene_name, CONFIG.panoptic_instance_cache_suffix_det);
            inst_feats = load(instance_path);
            instance_features_det = inst_feats.data;
            det_path = strcat(obj.dataset_path, obj.scene_name, '/detections.h5');
            bboxes_confs_h5 = cell(obj.nbr_frames, obj.nbr_cameras);
            for camera_idx = 1 : obj.nbr_cameras
                camera_name = obj.camera_names{camera_idx};
                for frame_idx = 1 : obj.nbr_frames
                    frame_name = obj.frame_names{frame_idx};
                    frame_name = strrep(frame_name, '.jpg', '');
                    key = strcat('/detections_', obj.scene_name, '_', camera_name, '_', frame_name);
                    bboxes_confs = h5read(det_path, key);
                    bboxes_confs_h5{frame_idx, camera_idx} = bboxes_confs;
                end
            end

            % Need to go from detection-instances to MubyNet instances
            obj.instance_features = cell(size(instance_features_det));
            for frame_idx = 1 : size(obj.instance_features, 1)
                for camera_idx = 1 : size(obj.instance_features, 2)

                    % Fetch everything for current frame-cam
                    preds2d = obj.pose_cache_2d{frame_idx, camera_idx};
                    inst_feats_det = instance_features_det{frame_idx, camera_idx};
                    dets = bboxes_confs_h5{frame_idx, camera_idx};

                    if isempty(inst_feats_det)
                        obj.instance_features{frame_idx, camera_idx} = inst_feats_det;
                        continue
                    end

                    % Compute IoUs
                    ious = nan(size(dets, 1), numel(preds2d));
                    for i = 1 : size(dets, 1)
                        det_bbox = dets(i, 1 : 4);
                        for j = 1 : numel(preds2d)
                            pred = preds2d{j};
                            confs = pred(:, 3);
                            pred = pred(:, 1 : 2);
                            pred(confs == 0, :) = nan;
                            pred_bbox = obj.pose_to_bbox(pred);
                            pred_bbox = [pred_bbox(3), pred_bbox(1), pred_bbox(4), pred_bbox(2)];
                            ious(i, j) = obj.helpers.iou(det_bbox, pred_bbox);
                        end
                    end

                    % Fetch instance features based on winning IoU matches
                    curr_inst_feats = -500 * ones(numel(preds2d), size(inst_feats_det, 2));
                    for i = 1 : numel(preds2d)
                        curr_ious = ious(:, i);
                        [iou, max_idx] = max(curr_ious);
                        if max(ious(max_idx, :)) == iou && iou >= CONFIG.panoptic_det_muby_min_iou
                            curr_inst_feats(i, :) = inst_feats_det(max_idx, :);
                        end
                    end
                    obj.instance_features{frame_idx, camera_idx} = curr_inst_feats;
                end
            end
        end

        function init_cached_camera_annots(obj)
            % Caches camera annotations (for speed)

            fprintf('init_cached_camera_annots\n');
            cache_path = strcat(obj.dataset_cache, '/', obj.scene_name, '_camera_annots_cache.mat');
            if exist(cache_path, 'file') == 2
                S = load(cache_path);
                obj.camera_annots_cache = S.camera_annots_cache;
            else
                fprintf('Camera annot cache not found -- creating!\n')
                obj.camera_annots_cache = cell(obj.nbr_persons, obj.nbr_cameras, obj.nbr_frames);
                for person_idx = 1 : obj.nbr_persons
                    for cam = 1 : obj.nbr_cameras
                        for f = 1 : obj.nbr_frames
                            obj.get_camera_annot(f, cam, person_idx);
                        end
                        fprintf(' for camera %d\n', cam);
                    end
                end
                camera_annots_cache = obj.camera_annots_cache;
                save(cache_path, 'camera_annots_cache');
            end
        end

        function joints = convert_joints(~, opj)
            % Expand from sparse, missing joints are set to nan
            if isstruct(opj)
                joints = nan(15, 3);
                for i = 1 : numel(opj)
                    joints(opj(i).id + 1, :) = [opj(i).xy', opj(i).conf];
                end
            else
                joints = opj;
            end

            % Convert to MPII
            joints = [
                joints(2, :);
                joints(1, :);
                0.5 * (joints(9, :) + joints(12, :));
                joints(6, :);
                joints(7, :);
                joints(8, :);
                joints(12, :);
                joints(13, :);
                joints(14, :);
                joints(3, :);
                joints(4, :);
                joints(5, :);
                joints(9, :);
                joints(10, :);
                joints(11, :)];
        end
        
        function load_predictor_cache(obj)
            % Caches (saves as mat-files) predictor 3D-pose predictions and
            % feature blob for all views
            global CONFIG

            % Path to blob cache dir
            cache_path = strcat(obj.dataset_cache, 'mubynet');
                
            % Create cache path if it doesn't exist
            if ~exist(cache_path, 'dir')
                mkdir(cache_path);
            end

            % Load joints
            fprintf('Trying to load pose cache...\n');
			joints_path_mat = strcat(cache_path, '/', obj.scene_name, '/', 'poses.mat');
			if exist(joints_path_mat, 'file') == 2

				% Load pose cache
				S = load(joints_path_mat);
				fieldnames_S = fieldnames(S);
				full_pose_cache = S.(fieldnames_S{1});

				% Separate into 2d and 3d pose caches
				obj.pose_cache_2d = full_pose_cache(:, :, 1);
				obj.pose_cache_3d = full_pose_cache(:, :, 2);
				
			   % Formatting
				for i = 1 : obj.nbr_frames
					for j = 1 : obj.nbr_cameras
						for k = 1 : numel(obj.pose_cache_2d{i, j})
							obj.pose_cache_2d{i, j}{k} = obj.convert_joints(obj.pose_cache_2d{i, j}{k});
						end
					end
				end
			end

			if ~(obj.predictor_cache_loaded || ~CONFIG.predictor_precache)

				% Init predictor cache struct
				obj.predictor_cache = struct();
				for i = 1 : numel(CONFIG.predictor_collect_blobs_from)
				   obj.predictor_cache.(CONFIG.predictor_collect_blobs_from{i}) = cell(obj.nbr_cameras, obj.nbr_frames);
				end

				% Load feature maps
				base_path = strcat(cache_path, '/', obj.scene_name, '/');
				blob_path = strcat(base_path, CONFIG.predictor_use_blob, '_small.mat');
				if exist(blob_path, 'file') == 2
					S = load(blob_path);
					fieldnames_S = fieldnames(S);
					obj.predictor_cache.(CONFIG.predictor_use_blob) = S.(fieldnames_S{1});
					obj.predictor_cache.(CONFIG.predictor_use_blob) = obj.predictor_cache.(CONFIG.predictor_use_blob)';
				else
					error('Only pre-cached predictor features implemented');
				end
			end
			if ~isempty(obj.pose_cache_3d)
				obj.init_detections();
			end

            % Go over pose cache and appropriately set empty views
            for f = 1 : size(obj.pose_cache_2d, 1)
                for c = 1 : size(obj.pose_cache_2d, 2)
                    nbr_poses = numel(obj.pose_cache_2d{f, c});
                    if nbr_poses == 1 && all(all(isnan(obj.pose_cache_2d{f, c}{1})))
                        obj.pose_cache_2d{f, c} = [];
                    end
                end
            end

            obj.predictor_cache_loaded = true;
            obj.check_data_consistency();
        end

        function unload_predictor_cache(obj)
            global CONFIG

            if ~obj.predictor_cache_loaded || ~CONFIG.predictor_precache
                % Return if cache is already loaded or caching is disabled.
                return;
            end

            obj.predictor_cache = struct;
            for i = 1 : numel(CONFIG.predictor_collect_blobs_from)
                obj.predictor_cache.(CONFIG.predictor_collect_blobs_from{i}) = ...
                    cell(obj.nbr_cameras, obj.nbr_frames);
            end
            obj.predictor_cache_loaded = false;
        end

        function [frame_annot, calib] = get_annot_calib(obj, frame_idx, person_idx)
            % This function returns all 3D information for a time-freeze,
            % e.g. for plotting of the 3D scene
            frame_annot = obj.frame_annots{person_idx, frame_idx};
            calib = obj.camera_calibrations;
        end

        function calibrations = load_camera_calibrations(obj)
            % This function reads the calibration-json file included with
            % the scene and makes transforms appropriately

            % Load json and extract relevant parts
            calib_path = strcat(obj.dataset_path, obj.scene_name);
            calib_json = strcat(calib_path, '/calibration_', obj.scene_name, '.json');
            calib_struct = json_decode(fileread(calib_json));
            calib_cell = calib_struct.cameras;

            % Convert to cell array because MATLAB is terrible inconsistent
            % when reading jsons.
            if ~iscell(calib_cell)
                c_cell = {};
                for i = 1:numel(calib_cell)
                    curr_calib = calib_cell(i);

                    if iscell(curr_calib.K)
                        curr_calib.K = reshape(cell2mat(curr_calib.K), [3,3])';
                    end

                    if iscell(curr_calib.R)
                        curr_calib.R = reshape(cell2mat(curr_calib.R), [3,3])';
                    end

                    if iscell(curr_calib.t)
                        curr_calib.t = cell2mat(curr_calib.t);
                    end

                   c_cell{end+1} = curr_calib;
                end
                calib_cell = c_cell;
            end


            idxs_to_remove = [];
            for i = 1 : numel(calib_cell)
                curr_calib_name = calib_cell{i}.name;
                if ~any(strcmp(obj.camera_names, curr_calib_name))
                    idxs_to_remove = [idxs_to_remove, i]; %#ok<*AGROW>
                end
            end

            calib_cell = calib_cell(setdiff(1 : numel(calib_cell), idxs_to_remove));

            % Insert into container
            calibrations = {};
            for camera_idx = 1 : numel(calib_cell)

                % Want similar indexing as for obj.cameras, i.e. we keep
                % the non-existing cell-entries as empty
                curr_calib = calib_cell{camera_idx};

                % Remove all irrelevant fields
                curr_calib = rmfield(curr_calib, 'name');
                curr_calib = rmfield(curr_calib, 'type');
                curr_calib = rmfield(curr_calib, 'resolution');
                curr_calib = rmfield(curr_calib, 'panel');
                curr_calib = rmfield(curr_calib, 'node');

                % Explicitly store the P-matrix for convenience
                curr_calib.P = curr_calib.K * [curr_calib.R, curr_calib.t];

                % Insert to container
                calibrations{end + 1} = curr_calib;
            end
        end

		function person_idx = gt2person(obj, gt_id)
			% gt_id is 0-based (from the raw Panoptic gt id format)
			% person_idx is 1-based (from the Matlab format)
		    person_idx = obj.body_id_2_idx(gt_id + 1);
		end

		function gt_id = person2gt(obj, person_idx)
			% person_idx is 1-based (from the Matlab format)
			% gt_id is 0-based (from the raw Panoptic gt id format)
		    gt_id = find(obj.body_id_2_idx == person_idx) - 1;
		end

        function projected_annot = get_projected_annot(obj, frame_idx, camera_idx, person_idx)
            % Get local (adapted to specific camera, indexed by camera_idx)
            % annotation for frame indexed by frame_idx

            % Extract global (frame-level) annotation
            [frame_annot, ~] = obj.get_annot_calib(frame_idx, person_idx);

            % Call frame2camera_pose function
            projected_annot = obj.project_frame_pose(frame_annot, camera_idx);
        end

        function projected_pose = project_frame_pose(obj, pose, camera_idx, already_rot_trans)
            % Given a 3D pose ('pose') in a global perspective (frame-level)
            % transform to the image-plane using the P-matrix
            % associated with camera indexed by camera_idx. The flag
            % already_rot_trans is true iff 'pose' has been / can already
            % be considered rotated and translated by R and t associated
            % with the camera (default: this is not assumed)

            % Default args
            if nargin < 4
                already_rot_trans = 0;
            end

            % Extract camera calibration info
            calib = obj.camera_calibrations{camera_idx};
            if already_rot_trans
                K = calib.K;
                P_pose_homo = K * (pose');
            else
                P = calib.P;
                pose_homo = [pose(:, 1 : 3), ones(size(pose, 1), 1)]';
                P_pose_homo = P * pose_homo;
            end
            P_pose_homo = bsxfun(@rdivide, P_pose_homo, P_pose_homo(end, :));
            projected_pose = P_pose_homo(1 : 2, :)';
        end

        function camera_annot = get_camera_annot(obj, frame_idx, camera_idx, person_idx)
            % This function rotates and translates the global (frame-level)
            % 3D pose annotation by the rotation matrix R and offset vector
            % t associted with the camera indexed by camera_idx, hence it
            % takes it into camera space

            % Check if the sought camera annot already is cached
            if isempty(obj.camera_annots_cache{person_idx, camera_idx, frame_idx})

                % Extract global (frame-level) annotation
                [frame_annot, ~] = obj.get_annot_calib(frame_idx, person_idx);

                % Extract camera calibration info
                calib = obj.camera_calibrations{camera_idx};
                R = calib.R;
                t = calib.t;

                % Apply rotation and translation
                if numel(frame_annot) > 0
                    frame_annot = frame_annot(:, 1 : 3)';
                    camera_annot = (R * bsxfun(@plus, frame_annot, t))';
                else
                    camera_annot = [];
                end
                obj.camera_annots_cache{person_idx, camera_idx, frame_idx} = camera_annot;
            else
                camera_annot = obj.camera_annots_cache{person_idx, camera_idx, frame_idx};
            end
        end

        function frame_perspective_coords = rotate_to_normal_view(obj, coords, camera_idx, R_extra)
            % This function rotates and translates to global (frame-level)

            % Extract camera calibration info
            calib = obj.camera_calibrations{camera_idx};
            R = calib.R;
            t = calib.t;

            if exist('R_extra', 'var')
                % Used to rotate from another coordinate system.
                R = R_extra * R;
            end

            % Apply rotation and translation
            coords = coords(:, 1 : 3)';
            frame_perspective_coords = (R' * bsxfun(@minus, coords, t))';
        end

        function camera_annot = rotate_to_camera(obj, frame_annot, camera_idx, R_extra)
            % This function rotates and translates to the camera perspective

            % Extract camera calibration info
            calib = obj.camera_calibrations{camera_idx};
            R = calib.R;
            t = calib.t;

            if exist('R_extra', 'var')
                R = R_extra * R;
            end

            frame_annot = frame_annot(:, 1 : 3)';
            camera_annot = (R * bsxfun(@plus, frame_annot, t))';
        end

        function [frame_annots, body_id_2_idx] = load_annotations(obj)
            % This function reads the annot-json files included with each
            % frame in the scene and transforms appropriately
            global CONFIG

            % Determine which person to track in case of a scene containing
            % multiple people.
            person_idx = nan;

            % Load json and extract relevant parts
            for frame_idx = 1 : obj.nbr_frames
                annot_name = obj.frame_names{frame_idx};
                annot_name = annot_name(1 : end - 4);
                annot_path = strcat(obj.dataset_path, obj.scene_name, '/', ...
                                    CONFIG.dataset_video_format, 'Pose3d_stage1');
                annot_json = strcat(annot_path, '/body3DScene_', annot_name, '.json');
                annot_struct = json_decode(fileread(annot_json));
                gt_bodies = annot_struct.bodies;
                if frame_idx == 1
                    % Find out nbr people (same in whole sequence)
                    obj.nbr_persons = numel(gt_bodies);
                    frame_annots = cell(obj.nbr_persons, obj.nbr_frames);
                    body_id_2_idx = [];
                    for body_idx = 1 : numel(gt_bodies)
                        body = gt_bodies(body_idx);
                        body_id_2_idx(body.id + 1) = body_idx;
                    end
                end

                % Note: Empty cell in frame_annots siginifices that no
                % annotation exists for that frame.
                if ~isempty(gt_bodies)

                    if isnan(person_idx)
                        % Select the first body as the person to track
                        % for now
                        person_idx = gt_bodies(1).id;
                    end

                    if numel(gt_bodies) ~= obj.nbr_persons
                        error('Number of people is in scene inconsistent!');
                    end

                    for body_idx = 1 : numel(gt_bodies)
                        body = gt_bodies(body_idx);
                        person_idx = body_id_2_idx(body.id + 1);
                        frame_annots{person_idx, frame_idx} = reshape(body.joints15, 4, 15)';
                    end
                else
                    error('Empty annotation: scene: %s, frame %d', obj.scene_name, frame_idx);
                end
            end
        end

        function pred_error = get_recon_error(obj, pred, frame_idx, person_idx, camera_idx)
            % This function computes the mean per-joint error in mm
            %
            % Input:
            %    - pred: 15 x 3 matrix, where each row is (x,y,z)
            %      prediction for the resp. joint
            %    - frame_idx: index of the frame
            %    - camera_idx (optional): index of the camera
            %
            % Output:
            %    - pred_error: Frobenius prediction error
            if exist('camera_idx', 'var')
                pred_errors = obj.get_recon_error_perjoint(pred, frame_idx, person_idx, camera_idx);
            else
                pred_errors = obj.get_recon_error_perjoint(pred, frame_idx, person_idx);
            end
            pred_error = mean(pred_errors);
        end

        function pred_errors = get_recon_error_perjoint(obj, pred, frame_idx, person_idx, camera_idx)
            % This function computes the mean per-joint error in mm
            %
            % Input:
            %    - pred: 15 x 3 matrix, where each row is (x,y,z)
            %      prediction for the resp. joint
            %    - frame_idx: index of the frame
            %    - camera_idx (optional): index of the camera
            %
            % Output:
            %    - pred_error: errors per joint

            global CONFIG

            % By default, assume pred is in frame-level perspective
            % (i.e. global view), as opposed to camera view
            if ~exist('camera_idx', 'var')
                % Global (panoptic) perspective
                annot = obj.frame_annots{person_idx, frame_idx};
            else
                % Rotated camera perspective
                % Check if the recon error exists in cache
                annot = obj.get_camera_annot(frame_idx, camera_idx, person_idx);
            end
            % Hip-align
            pred = obj.helpers.trans_hip_coord(pred, annot(:, 1 : 3));
            pred_errors = sqrt(sum((pred - annot(:, 1 : 3)).^2, 2)) / ...
                            CONFIG.panoptic_scaling_factor;
        end

         function reproj_error = get_reproj_error(obj, cum_pred, fr_idx, person_idx)

            % Iterate over all camera's 2d projected gts and compute
            % reprojection error to these
            reproj_error = 0;
            nbr_actual_dets = 0;
            for other_camera_id = 1 : obj.nbr_cameras

                % Rotate and project to current camera
                cum_pred_2d = obj.project_frame_pose(cum_pred, other_camera_id, 0);

                % Get current pose index
                [pose_idx, ~] = obj.get_gt_detection(fr_idx, other_camera_id, person_idx);

                if pose_idx ~= -1
                    nbr_actual_dets = nbr_actual_dets + 1;
                    gt_2d = obj.get_projected_annot(fr_idx, other_camera_id, person_idx);
                    reproj_error = reproj_error + ...
                                    mean(sqrt(sum((cum_pred_2d - gt_2d).^2, 2)));
                end
            end
            if nbr_actual_dets > 0
                reproj_error = reproj_error / nbr_actual_dets;
            end
        end

        function global_angles = global_angles_cam(obj, camera_idx, to_individual)
            if ~exist('to_individual', 'var')
                to_individual = 1;
            end
            if to_individual
                global_angles = obj.camera_rig_individual.global_angles_cam(camera_idx);
            else
                global_angles = obj.camera_rig.global_angles_cam(camera_idx);
            end
        end

        function init_pred_ordering(obj)
            if isempty(obj.pose_cache_3d)
                return;
            end

            fprintf('Ordering best camera indexes for scene %s\n', obj.scene_name);

            % Container of all best orderings, which has nbr_persons + 1,
            % here (nbr_persons + 1)th entry is average over all persons
            best_cam_ord_curr = nan(obj.nbr_persons + 1, obj.nbr_frames, ...
                                    obj.nbr_cameras);
            
            % Iterate over all frames
            for fr_idx = 1 : obj.nbr_frames

                % Iterate over all persons
                curr_errors = nan(obj.nbr_persons, obj.nbr_cameras);
                for person_idx = 1 : obj.nbr_persons

                    % Iterate over all cameras
                    errors = nan(obj.nbr_cameras, 1);
                    for camera_id = 1 : obj.nbr_cameras

                        % Get detection best matching person
                        [det_id, ~] = obj.get_gt_detection(fr_idx, camera_id, person_idx);

                        % Run predictor on this view
                        [pred, ~] = obj.get_predictor(fr_idx, camera_id, det_id);
                        annot = obj.get_camera_annot(fr_idx, camera_id, person_idx);
                        pred = obj.helpers.trans_hip_coord(pred, annot);

                        % Compute error
                        pred_zeros = pred;
                        pred_zeros(isnan(pred_zeros)) = 0;
                        errors(camera_id) = norm(pred_zeros - annot);
                    end
                    curr_errors(person_idx, :) = errors;
                end

                % Sort from best to worst (averaged over all people)
                [~, best_inds] = sort(sum(curr_errors, 1), 'ascend');
                best_cam_ord_curr(obj.nbr_persons + 1, fr_idx, :) = best_inds;
                for person_idx = 1 : obj.nbr_persons
                    [~, best_inds] = sort(curr_errors(person_idx, :), 'ascend');
                    best_cam_ord_curr(person_idx, fr_idx, :) = best_inds;
                end
            end
            obj.best_cam_ord = best_cam_ord_curr;
        end
    end
end
