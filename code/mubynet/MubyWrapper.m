classdef MubyWrapper
    % Wrapper around the MubyNet model
    
    properties
        img_size
        param_sampling
        net_sampling
        param_scoring
        net_scoring
    end

    methods
        
        function obj = MubyWrapper()
            
            global CONFIG

            % Disable MubyNet?
            if CONFIG.predictor_disabled
                obj.net_sampling = 0;
                return
            end

            % Set defaults from MubyNet paper (as suggested by the authors)
            obj.img_size = [300, nan];
            
            % Add paths
            addpath(genpath('./code/mubynet/'));
            
            % Set gpu options and load caffe model
            obj.param_sampling = config_release(1);
            obj.net_sampling = caffe.Net(CONFIG.muby_deploy_proto_sampling, ...
                                         CONFIG.muby_weights, 'test');
            obj.param_scoring = config_release(2);
            obj.net_scoring = caffe.Net(CONFIG.muby_deploy_proto_scoring, ...
                                        CONFIG.muby_weights, 'test');
        end
        
        function [pose2D, pose3D, translation, feature_blobs] = predict(obj, img, calib)
            
            % Default args
            if ~exist('calib', 'var')
                calib = nan;
            end
            
            if obj.net_sampling == 0
               % if MubyNet is disabled
                error('Called MubyWrapper.predict on when MubyNet is disabled');
            end
            
            % Use MubyNet module for 3D pose prediction
            height_ratio = size(img, 1) / obj.img_size(1);
            img = imresize(img, obj.img_size);
            
            % sampling and grouping
            final_score = applyModel_3d(img, obj.param_scoring, obj.net_sampling, 1);
            
            % BIP decoding
            [~, ~, ~, limbs3d, limbs2d] = BIP_decode(final_score, obj.param_scoring, obj.net_scoring);
            
            % Estimate pose translations (not used)
            translation = nan;
            
            % Re-format output
            pose2D = limbs2d;
            if ~isempty(pose2D)
                
                % Reshape arrays
                pose2D(:, 1 : 2, :) = height_ratio * pose2D(:, 1 : 2, :);
                pose2D = permute(pose2D, [3, 2, 1]);
                pose3D = permute(limbs3d, [3, 2, 1]);
                
                % Make into N-by-1 cell where N = number of poses
                pose2D_cell = cell(size(pose2D, 3), 1);
                pose3D_cell = cell(size(pose2D_cell));
                for k = 1 : numel(pose2D_cell)
                    pose2D_cell{k} = pose2D(:, :, k);
                    pose3D_cell{k} = pose3D(:, :, k);
                end
                pose2D = pose2D_cell;
                pose3D = pose3D_cell;
            else
                pose2D = {};
                pose3D = {};
            end
                
            % Extract feature map(s) from network
            feature_blobs = struct;
            blob_name = 'conv4_4_CPM';
            blob_id = obj.net_sampling.name2blob_index(blob_name);
            feature_blobs.(blob_name) = imresize(obj.net_sampling.blob_vec(blob_id).get_data(), 0.5);
        end
    end  
end
