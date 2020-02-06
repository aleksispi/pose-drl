function run_generate_finetuned_instance_cache(instance_weights, mode, vgg_path, fine_itr)
% Generates a cache of instance features for the detections

if ~exist('mode', 'var')
    mode = 'train';
end

clc;
close all;

% Add paths
addpath('code/');
addpath('code/panoptic');
addpath('code/utils');

% Setup global config settings
load_config(mode);

% Create helper
helper = Helpers();
helper.setup_caffe();

% Launch panoptic environment and start caching
global CONFIG
CONFIG.predictor_limited_caching = 1;
CONFIG.predictor_disabled = 1;
CONFIG.predictor_precache = 0;

vgg = caffe.Net('models/instance_classifier/vgg19.prototxt', vgg_path, 'test');
env = Panoptic(CONFIG.dataset_path, CONFIG.dataset_cache, nan);

for scene_idx = 1:numel(env.scenes)
    env.goto_scene(scene_idx);
    fprintf('Scene: %s\n', env.scene().scene_name);
    save_path = strcat(CONFIG.dataset_cache, '/', 'mubynet', '/', env.scene().scene_name, '/instance_tuned.mat');
    if exist(save_path, 'file') == 2
        fprintf('Already cached, skipping!\n');
        continue
    end

    % Reset network before fine tuning
    if env.scene().nbr_persons > 1
        fined_tuned_weights = sprintf('%s%s_instance_weights_gt_%d.caffemodel', CONFIG.dataset_cache, env.scene().scene_name, fine_itr);
        
        if exist(fined_tuned_weights, 'file') ~= 2
            fprintf('Fine tuning instance network...\n');
            solver = caffe.get_solver('models/instance_classifier/solver.prototxt');
            net = solver.net;
            net.copy_from(instance_weights);
            finetune(solver, net, vgg, env, fine_itr);
            net.save(fined_tuned_weights);
        else
            fprintf('Fine tuned weights already found, using those!\n');
        end
        net = caffe.Net('models/instance_classifier/deploy.prototxt', fined_tuned_weights, 'test');
    else
        fprintf('Skip fine tuning, not multiple people!\n');
        net = caffe.Net('models/instance_classifier/deploy.prototxt', instance_weights, 'test');
    end

    data = cell(env.scene().nbr_frames, env.scene().nbr_cameras);
    for frame_idx = 1:env.scene().nbr_frames
        fprintf('  frame: %d/%d\n', frame_idx, env.scene().nbr_frames);
        for camera_idx = 1:env.scene().nbr_cameras
            poses = env.scene().pose_cache{frame_idx, camera_idx};
			bboxes_dataset = strcat('/detections_', env.scene().scene_name, '_', ...
												 env.scene().camera_names{camera_idx}, '_', ...
												 strrep(env.scene().frame_names{frame_idx}, '.jpg', ''));                                     
			det_path = strcat(env.scene().dataset_path, env.scene().scene_name, '/detections.h5');
			det_boxes = h5read(det_path, bboxes_dataset);
            nbr_poses = numel(poses);
            
            % Get Faster R-CNN detection boxes
            detections = nan(nbr_poses, 50);
            for detection_id = 1:numel(poses)
                
				pred = poses{detection_id};
				confs = pred(:, 3); 
				pred = pred(:, 1 : 2);
				pred(confs == 0, :) = nan;
				muby_box = env.scene().pose_to_bbox(pred);
				
				% try to match mubynet pose with best faster-rcnn box
				best_iou = 0;
				best_faster_box = nan;
				for faster_id = 1:size(det_boxes, 1)
					fast_box = [det_boxes(faster_id, 2), det_boxes(faster_id, 4), det_boxes(faster_id, 1), det_boxes(faster_id, 3)];
					iou = helper.iou(fast_box, muby_box);
					if iou > best_iou
						best_iou = iou;
						best_faster_box = fast_box;
					end
				end
				
				if best_iou > 0.1
					bbox = best_faster_box;
				else
					bbox = muby_box;
				end

                img = env.scene().get_img(frame_idx, camera_idx);

                % Get bbox
                img = env.scene().crop_human(img, bbox);
                
                % Get VGG19 features
                img = imresize(img, [224, 224]);
                vgg.blobs('data').set_data(img);
                vgg.forward_prefilled();
                f = vgg.blobs('conv5_4/bn').get_data();
                f = f(:);

                % Get instance features
                net.blobs('data').set_data(f);
                net.forward_prefilled();
                f1 = net.blobs('feat').get_data();
                detections(detection_id, :) = f1;
            end
            data{frame_idx, camera_idx} = detections;
        end
    end
    save(save_path, 'data');
end
end

function finetune(solver, net, vgg, env, iterations)

    total_time = tic;
    training_time = 0;

    batch_size = 16;
    for it = 1:iterations

        fprintf('Batch %d/%d\n', it, iterations);
        % Build batch
        batch = cell(batch_size, 2);
        for batch_idx = 1:batch_size

            % Either matching pair or not.
            if mod(batch_idx, 2) == 0
                same = 1;
            else
                same = 0;
            end

            [f1, f2, t] = get_pair(env, same, vgg);
            training_time = training_time + t;

            batch{batch_idx, 1} = f1;
            batch{batch_idx, 2} = f2;
            batch{batch_idx, 3} = same;
        end

        tic
        net.blobs('data').set_data(cat(2, batch{:, 1}));
        net.blobs('data_p').set_data(cat(2, batch{:, 2}));
        net.blobs('label').set_data(cat(2, batch{:, 3}));
        training_time = training_time + toc;

        % Do training
        solver.step(1);
        loss = net.blobs('loss').get_data();

        if mod(it, 10) == 0
            fprintf('Training steps: %d, samples: %d, loss: %f\n', it, it * batch_size, loss);
        end

    end

    fprintf('Total refine time: %ds, time spent training: %ds\n', toc(total_time), training_time);

end



%%% Start of training functions from run_train_instance_detector.m

function [f1, f2, vgg_time] = get_pair(env, same, vgg)
    % Random scene
    while 1
        vgg_time = 0;

        % Random start frame, camera, scene, person
        % env.reset();  --- do not change scene
        env.goto_frame(randi(env.scene().nbr_frames));
        env.goto_cam(randi(env.scene().nbr_cameras));
        env.goto_person(randi(env.scene().nbr_persons));

        % Retry until good view
        [yes, ~] = is_visible(env);
        if ~yes
            continue
        end

        % Get features for person 1
        [f1, t] = get_features(env, vgg);
        vgg_time = vgg_time + t;

        % Switch target if negative example
        if ~same
            old_pid = env.person_idx;
            new_pid = old_pid;
            while new_pid == old_pid
               env.goto_person(randi(env.scene().nbr_persons));

               new_pid = env.person_idx;
            end
        end

        frame_idx = env.frame_idx;
        tries = 5;
        while tries > 0
            % Prevent getting stuck in a loop
            tries = tries -1;

            % Select new random view
            max_frame = min(frame_idx + 5 * 10, env.scene().nbr_frames);
            min_frame = max(frame_idx - 5 * 10, 1);

            env.goto_frame(randi([min_frame, max_frame]));
            env.goto_cam(randi(env.scene().nbr_cameras));

            % Retry until good view
            [yes, ~] = is_visible(env);
            if yes
                break
            end
        end

        if tries <= 0
            % Didn't find a good f2, restart with new f1.
            continue;
        end
        [f2, t] = get_features(env, vgg);
        vgg_time = vgg_time + t;

        break;

    end
end

function [f, t] = get_features(env, vgg)

    img = get_features_img(env);
    img = imresize(img, [224, 224]);
    vgg.blobs('data').set_data(img);
    tic;
    vgg.forward_prefilled();
    f = vgg.blobs('conv5_4/bn').get_data();
    t = toc;
    f = f(:);

end

% Function to extract features
function f = get_features_img(env)
    f = env.get_current_img();

    global CONFIG

    annots = env.scene().get_projected_annot(env.frame_idx, env.camera_idx, env.person_idx);
    annots = round(annots);
    coord_min = min(annots);
    coord_max = max(annots);
    coord_min(1) = coord_min(1) - CONFIG.panoptic_crop_margin(1);
    coord_min(2) = coord_min(2) - CONFIG.panoptic_crop_margin(2);
    coord_max(1) = coord_max(1) + CONFIG.panoptic_crop_margin(1);
    coord_max(2) = coord_max(2) + CONFIG.panoptic_crop_margin(2);
    bbox = [coord_min(2), coord_max(2), coord_min(1), coord_max(1)];

    f = env.scene().crop_human(f, bbox);
end

function bbox = pose_to_bbox(pose)
    p_start = min(pose);
    p_size = max(pose) - p_start;
    bbox = [p_start(1), p_start(2), p_size(1), p_size(2)];
end

% Function to check if the target is visible and not occluded.
function [yes, bbox_idx] = is_visible(env)
    yes = 0;
    bbox_idx = -1;

    pose = env.scene().get_projected_annot(env.frame_idx, env.camera_idx, env.person_idx);
    bbox = pose_to_bbox(pose);

    if bbox(3) < 32 || bbox(4) < 32
        return;
    end

    img_box = [0, 0, 1920, 1080];

    % bbox overlaps with image
    in_ratio = bboxOverlapRatio(bbox, img_box, 'min');

    if in_ratio < 0.8
        return
    end

    old_pid = env.person_idx;
    for pid = 1:env.scene().nbr_persons
        if pid ~= old_pid
            env.goto_person(pid);
            pose = env.scene().get_projected_annot(env.frame_idx, env.camera_idx, env.person_idx);
            otherbox = pose_to_bbox(pose);
            ratio = bboxOverlapRatio(bbox, otherbox, 'union');

            if ratio > 0.20
                env.goto_person(old_pid);
                return
            end

        end
    end
    env.goto_person(old_pid);
    
    yes = 1;
    return;
end
