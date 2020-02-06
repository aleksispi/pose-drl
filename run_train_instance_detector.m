function run_train_instance_detector(mode, vgg_path)
% Trains an instance detector

if ~exist('mode', 'var')
    mode = 'train';
end

MODEL_NAME=''

clc;
close all;

% Add paths
addpath(genpath('code/'));

% Setup global config settings
load_config(mode);

% Create helper
helper = Helpers();
helper.setup_caffe();
predictor = nan;

% Launch panoptic environment and start caching
global CONFIG
CONFIG.predictor_limited_caching = 1;
CONFIG.predictor_disabled = 1;
CONFIG.predictor_precache = 0;

solver = caffe.get_solver('models/instance_classifier/solver.prototxt');
net = solver.net;

env_train = Panoptic(CONFIG.dataset_path, CONFIG.dataset_cache, predictor);
env_val = Panoptic(strrep(CONFIG.dataset_path, 'train', 'val'), strrep(CONFIG.dataset_cache, 'train', 'val'), predictor);

vgg = caffe.Net('models/instance_classifier/vgg19.prototxt', vgg_path, 'test');

iterations = 100000;
batch_size = 16;

bad_imgs = 0;

for it = 1:iterations

    val = mod(it, 1000) == 0;
    
    % Build batch
    batch = cell(batch_size, 2);
    for batch_idx = 1:batch_size
        
        % Either matching pair or not.
        if mod(batch_idx, 2) == 0
            same = 1;
        else
            same = 0;
        end
        
        if val
            [f1, f2] = get_pair(env_val, same, vgg);
        else
            [f1, f2] = get_pair(env_train, same, vgg);
        end
        
        batch{batch_idx, 1} = f1;
        batch{batch_idx, 2} = f2;
        batch{batch_idx, 3} = same;   
    end
    
    net.blobs('data').set_data(cat(2, batch{:, 1}));
    net.blobs('data_p').set_data(cat(2, batch{:, 2}));
    net.blobs('label').set_data(cat(2, batch{:, 3}));
    
    if val
        % Every 1000 step evaluate the model

        net.forward_prefilled();
        f1 = net.blobs('feat').get_data();
        f2 = net.blobs('feat_p').get_data();
        same = net.blobs('label').get_data();
        
        % Mean distance for same person
        dist = sqrt(sum((f1-f2)'.^2, 2));
        d_same = mean(same' .* dist);
        d_diff = mean((same == 0)' .* dist);
        
        fprintf('Training steps: %d, samples: %d, d_same: %f, d_diff: %f, bad_imgs: %d\n', it, it * batch_size, d_same, d_diff, bad_imgs);
        model_path = sprintf('models/instance/model-%s_%d-itr_%s.caffemodel\n', MODEL_NAME, it,  datestr(now, 'yyyy-mm-dd-HH-MM-SS'));
        net.save(model_path);
        fprintf('Saving model to: %s', model_path);
    else  
       % Do training
        solver.step(1);
        loss = net.blobs('loss').get_data();   
       
        if mod(it, 10) == 0
            fprintf('Training steps: %d, samples: %d, loss: %f\n', it, it * batch_size, loss);
        end
    end

end
end

function [f1, f2] = get_pair(env, same, vgg)
    % Random scene
    while 1
        
        % Random start frame, camera, scene, person
        env.reset();
        env.goto_person(randi(env.scene().nbr_persons));

        % Retry until good view
        [yes, bbox_idx] = is_visible(env);
        if ~yes
            continue
        end
        
        % Get features for person 1
        f1 = get_features(env, vgg);

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
            [yes, bbox_idx] = is_visible(env);
            if yes
                break
            end
        end
        
        if tries <= 0
            % Didn't find a good f2, restart with new f1.
            continue;
        end
        f2 = get_features(env, vgg);
        
        break;
        
    end
end

function f = get_features_faster_rcnn(env, bbox_idx)
    fcs_dataset = strcat('/fcs_', env.scene().scene_name, '_', ...
                                         env.scene().camera_names{env.camera_idx}, '_', ...
                                         strrep(env.scene().frame_names{env.frame_idx}, '.jpg', ''));                                     
    det_path = strcat(env.scene().dataset_path, env.scene().scene_name, '/detections.h5');
    data = h5read(det_path, fcs_dataset);
    f = data(bbox_idx, :)';
end

function f = get_features(env, vgg)

    img = get_features_img(env);
    img = imresize(img, [224, 224]);
    vgg.blobs('data').set_data(img);
    vgg.forward_prefilled();
    f = vgg.blobs('conv5_4/bn').get_data();
    f = f(:);

end

% Function to extract features
function f = get_features_img(env)
    %[~, f] = env.get_current_predictor();
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
    for pid = 1:env.scene().nbr_persons;
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
    
    % Check for overlapping detection box
    bboxes_dataset = strcat('/detections_', env.scene().scene_name, '_', ...
                                         env.scene().camera_names{env.camera_idx}, '_', ...
                                         strrep(env.scene().frame_names{env.frame_idx}, '.jpg', ''));                                     
    det_path = strcat(env.scene().dataset_path, env.scene().scene_name, '/detections.h5');
    data = h5read(det_path, bboxes_dataset);
    max_ratio = 0;
    max_idx = 0;

    for box_idx = 1:size(data, 1)
        % Crop  dection to get only human
        det_box = [data(box_idx, 1), data(box_idx, 2), data(box_idx, 3) - data(box_idx, 1), data(box_idx, 4) - data(box_idx, 2)];
        ratio = bboxOverlapRatio(bbox, det_box, 'union');
        
        if ratio > max_ratio
           max_ratio = ratio;
           max_idx = box_idx;
        end
    end

    if max_ratio > 0.5
        yes = 1;
        bbox_idx = max_idx;
    end

end
