classdef Helpers < handle
    
    methods
        function [files, nbr_files] = list_filenames(obj, path) %#ok<*INUSL>
            % Lists and counts all files in a directory
            files = dir(path);
            files = struct2cell(files);
            files = files(1, 3 : end);
            nbr_files = numel(files);
        end
        
        function setup_caffe(~)
            global CONFIG
            % WARNING! Adjust the path to your caffe accordingly!
            caffepath = './caffe/matlab';
            fprintf('You set your caffe in caffePath.cfg to: %s\n', caffepath);
            addpath(caffepath);
            caffe.reset_all();
            if CONFIG.use_gpu
                caffe.set_mode_gpu();
                caffe.set_device(CONFIG.gpu_id);
            else
                caffe.set_mode_cpu();
            end
        end
        
        function a = plot3d_and_cams(obj, env, preds, title_str, ...
                                     legend_str, new_fig, colors, ...
                                     show_all_cams, show_err_heatmap, ...
                                     show_top_prct, show_plot)
            if ~iscell(preds); preds = {preds}; end
            if ~exist('new_fig', 'var'); new_fig = 1; end
            if ~exist('colors', 'var')
                if numel(preds) == 1
                    colors = {'b'};
                else
                    colors = {'b', 'r'};
                end
            elseif ~iscell(colors); colors = {colors};
            end
            if ~exist('show_all_cams', 'var'); show_all_cams = 1; end
            if ~exist('show_err_heatmap', 'var'); show_err_heatmap = 0; end
            if ~exist('show_top_prct', 'var'); show_top_prct = 1.0; end
            if ~exist('show_plot', 'var'); show_plot = 1; end
            if new_fig
                if show_plot
                    a = figure;
                else
                    a = figure('visible', 'off');
                end
            else
                a = nan;
            end
            hold on;
            for i = 1 : numel(preds)
                pred = preds{i};
                color = colors{i};
                if size(pred, 1) == 15
                    obj.plot_skeleton(pred(:, 1:3), color);
                else
                    plot3(pred(:, 1), pred(:, 2), pred(:, 3), '*');
                end
                axis equal; xlabel('x'); ylabel('y'); zlabel('z');
            end
            title(title_str);
            if numel(legend_str) > 0
                legend(legend_str)
            end
            grid on;
            env.scene().camera_rig.show_cam_rig(0, show_all_cams);
        end
        
        function set_agent_m_values(~)
            % Want linear scheme for increasing m. Therefore we first
            % calculate how many batch gradient updates will be in total
            global CONFIG
            
            % Initialize container of m-parameter updates (values)
            nbr_batch_updates = floor(CONFIG.training_agent_nbr_eps / ...
                                      CONFIG.training_agent_eps_per_batch);
            CONFIG.agent_m_values = nan(2, nbr_batch_updates);
            agent_ms_all = CONFIG.agent_ms;
            for i = 1 : numel(agent_ms_all)
                agent_ms = agent_ms_all{i};
                ms_start = agent_ms{1};
                ms_end = agent_ms{2};
                interval = agent_ms{3};
                if numel(interval) == 1
                    % Attach end of training episode to interval
                    interval = [interval, CONFIG.training_agent_nbr_eps];
                end
                curr_nbr_batch_updates = ceil((interval(2) - interval(1) + 1) / ...
                                               CONFIG.training_agent_eps_per_batch);
                interval = ceil(interval / CONFIG.training_agent_eps_per_batch);
                curr_azims = linspace(ms_start(1), ms_end(1), curr_nbr_batch_updates);
                curr_elevs = linspace(ms_start(2), ms_end(2), curr_nbr_batch_updates);
                CONFIG.agent_m_values(:, interval(1) : interval(2)) = ...
										[curr_azims; curr_elevs];
            end
            
            % Fill in the m-constant parts
            for i = 1 : nbr_batch_updates
				if isnan(CONFIG.agent_m_values(1, i))
					CONFIG.agent_m_values(:, i) = CONFIG.agent_m_values(:, i - 1);
				end
            end
        end
        
        function [n,V,p] = affine_fit(obj, X)
            %Computes the plane that fits best (lest square of the normal distance
            %to the plane) a set of sample points.
            % source: https://www.mathworks.com/matlabcentral/fileexchange/43305-plane-fit
            %INPUTS:
            %
            %X: a N by 3 matrix where each line is a sample point
            %
            %OUTPUTS:
            %
            %n : a unit (column) vector normal to the plane
            %V : a 3 by 2 matrix. The columns of V form an orthonormal basis of the
            %plane
            %p : a point belonging to the plane
            %
            %NB: this code actually works in any dimension (2,3,4,...)
            %Author: Adrien Leygue
            %Date: August 30 2013

            %the mean of the samples belongs to the plane
            p = mean(X,1);

            %The samples are reduced:
            R = bsxfun(@minus,X,p);
            %Computation of the principal directions if the samples cloud
            [V, ~] = eig(R'*R);
            %Extract the output from the eigenvectors
            n = V(:,1);
            V = V(:,2:end);
        end

        function plot_skeleton(obj, pose, jointColor, joint_size, lineWidth, perspective)
            % pose3D - 15x3 matrix of 3D joints or 15x2 for 2d joints
            % colorOption - color to plot (default='r')
            if(nargin <= 2)
                jointColor = 'r';
            end
            if(nargin <= 3)
                joint_size = 100;
            end
            if(nargin <= 4)
                lineWidth = 5;
            end
            if(nargin <= 5)
                perspective = [-3, -56];
            end
            
            hold on;
            is_3d = size(pose, 2) >= 3;
            
            % configure plot
            set(gcf,'Color',[1,1,1]);
            axis equal;
            ax = gca;               % get the current axis
            ax.Clipping = 'off';    % turn clipping off
            axis off;
            grid on;
            
            if is_3d
                % only rotate image if we show 3d plot
                view(perspective);
            end
            
            neck = 1;
            head = 2;
            center = 3;
            lshoulder = 4;
            lelbow = 5;
            lwrist = 6;
            lhip = 7;
            lknee = 8;
            lankle = 9;
            rshoulder = 10;
            relbow = 11;
            rwrist = 12;
            rhip = 13;
            rknee = 14;
            rankle = 15;
            
            order = [1 2 3];
                        
            connections = [ 
                head neck;            
                neck center;
                
                lshoulder neck;
                rshoulder neck;
                
                lshoulder lelbow;
                lelbow lwrist;
                
                rshoulder relbow;
                relbow rwrist;
                
                rhip rknee;
                rknee rankle;
                
                lhip lknee;
                lknee lankle;     
                
                lhip center;
                rhip center;
            ];

           
            pose = pose';
            % plot limbs
            for i = 1:size(connections, 1)
                if any(isnan(pose(:, connections(i, :))))
                    % Skip nan joints
                    continue;
                end
                c = jointColor;
                if is_3d
                    plot3(pose(order(1), connections(i, :)), pose(order(2), connections(i, :)), ...
                          pose(order(3), connections(i, :)), '-', 'Color', c, 'LineWidth', lineWidth);          
                else
                    plot(pose(order(1), connections(i, :)), pose(order(2), connections(i, :)), ...
                         '-', 'Color', c, 'LineWidth', lineWidth);
                end
            end
            
            pose = pose';
            % plot joints
            colors = parula(size(pose, 1));
            if is_3d
                scatter3(pose(:, order(1)), pose(:, order(2)), pose(:, order(3)), ...
                         joint_size, colors, 'filled');
            else
                scatter(pose(:, order(1)), pose(:, order(2)), ...
                        joint_size, colors, 'filled');
            end
        end
        
        function plot_ground_plane(obj, poses, color, size)
            if(nargin <= 2)
                color = 'red';
            end
            if(nargin <= 3)
                size = 200;
            end
            
            % Takes a cell array of 3d poses
            rankle = 15;
            lankle = 9;
            feet = [];
            for i = 1:numel(poses)
                pose = poses{i};
                f = pose([rankle, lankle], :);
                f = f(~isnan(f(:, 1)), :);
                feet = [feet; f];
            end
            
            if numel(feet) > 0
                % We want to make sure the axis isn't affected by plane
                h = gca;
                hold on
                h.YLimMode='manual';
                h.XLimMode='manual';
                h.ZLimMode='manual';
                
                % Fit plane
                [~, V_2, p_2] = obj.affine_fit(feet);
                steps = -1:1;
                num_pts = numel(steps);
                [S1,S2] = meshgrid(steps * size);
                %generate the pont coordinates
                X = p_2(1)+[S1(:) S2(:)]*V_2(1,:)';
                Y = p_2(2)+[S1(:) S2(:)]*V_2(2,:)';
                Z = p_2(3)+[S1(:) S2(:)]*V_2(3,:)';
                %plot the plane
                s = surf(reshape(X,num_pts, num_pts),reshape(Y, num_pts, num_pts),reshape(Z, num_pts, num_pts),'facecolor', color,'facealpha', 0.1);
                %s.EdgeColor = [0.3, 0.3, 0.3];
                %set(h, 'LineWidth', 0.8);
                s.EdgeColor = 'none';
            end
        end
        
        function fig = plot_smpl(obj, J_predictor, hip_center, is_male, color, fig)
            % Plots SMPL into the current figure
            if ~isobject(fig)
               fig = figure('visible', 'off'); 
            end
            hold on;

            opts.ismale = is_male; % what SMPL mesh to use -- female / male
            opts.color = color;
            opts.T = hip_center / 1000; % model translation in camera space
            
            J_predictor = double(J_predictor);
            prettyViewer('/', J_predictor, opts);
            
            hold off;
        end
        
        function c = color_from_pid(obj, pid, max_people)
            % Generates a color from a map that is consistent with pid          
            colors = parula(max_people);
            color = colors(pid, :);

            % Convert from RGB to HSL, increase light then back to RGB
            color = obj.rgb2hsl(color);
            color(3) = color(3) * 1.50;
            c = obj.hsl2rgb(color);
        end
        
        function time_freezes = get_time_freezes(obj, env, mode)
            global CONFIG
            
            time_freezes = [];
            counter = 0;
            
            for i = 1 : numel(env.scenes)
               
                scene_name = strcat('scene_', env.scenes{i}.scene_name);
                
                if strcmp(mode, 'train')
                    sequence_length = CONFIG.sequence_length_train;
					if ~isempty(strfind(scene_name, 'pose'))
						frame_step_length = max(1, sequence_length - ceil(0.5 * sequence_length));
					elseif ~isempty(strfind(scene_name, 'mafia')) 
						frame_step_length = max(1, sequence_length - ceil(0.5 * sequence_length));
					elseif ~isempty(strfind(scene_name, 'ultimatum')) 
						frame_step_length = max(1, sequence_length - ceil(0.5 * sequence_length));
					else
						frame_step_length = 1;
					end
                else
                    sequence_length = CONFIG.sequence_length_eval;
					frame_step_length = 1;
                end
                if frame_step_length > 1
                    % used so that each time we may get slightly different
                    % data shuffle
                    if ~isfield(CONFIG, 'trainset_seed')
                        CONFIG.trainset_seed = 0;
                    else
                        CONFIG.trainset_seed = CONFIG.trainset_seed + 1;
                    end
                    rng(CONFIG.trainset_seed);
                    offset = randi(10) - 1;
                    rng(CONFIG.rng_seed);
                else
                    offset = 0;
                end
                
                for j = 1 + offset : frame_step_length : env.scenes{i}.nbr_frames - CONFIG.sequence_step * (sequence_length - 1)
                    % If quality of time_freeze is good enough, then we
                    % will allow it to be part of the training set
					ncams = env.scenes{i}.nbr_cameras;
					time_freezes = [time_freezes; ...
									[i * ones(ncams, 1), ...
									j * ones(ncams, 1), ...
									(1 : ncams)', ...
									nan(ncams, 1)]];
                    counter = counter + 1;                   
                end
            end
        end
            
        % Below functions useful for chaning prototxt files
        function proto_name = set_proto_name(~, type)
            global CONFIG
            proto_name = strcat(CONFIG.agent_proto_folder, type);
            if numel(CONFIG.agent_proto_name) > 0
                proto_name = strcat(proto_name, '_', CONFIG.agent_proto_name);
            end
            proto_name = strcat(proto_name, '_', 'mubynet', '.prototxt');
        end
        
        function set_solver_proto(obj)
            
            global CONFIG
            
            % Read and get interesting lines (and then also close orig file)
            file_id = fopen(CONFIG.agent_solver_proto);
            [file_content, idxs_nonempty_noncommented] = obj.read_line_by_line(file_id);

            % Change values of desired fields
            fields_to_process = {'base_lr', 'gamma', 'random_seed',  'lr_policy'};

            new_vals = {CONFIG.training_agent_lr, CONFIG.training_agent_lr_update_factor, ...
                        CONFIG.rng_seed_caffe, sprintf('"%s"', CONFIG.caffe_lr_policy)};

            for i = 1 : numel(fields_to_process)
                field = fields_to_process{i};
                id = find(~cellfun(@isempty, strfind(file_content, field)));
                id = id(idxs_nonempty_noncommented(id)); id = id(end);
                line = file_content{id};
                line1_split = strsplit(line, ':');
                new_val = new_vals{i};
                new_line = strcat(line1_split{1}, ':', {' '}, num2str(new_val));
                file_content{id} = new_line{1};
            end

            % Replace in solver.prototxt
            file_id = fopen(CONFIG.agent_solver_proto, 'w');
            obj.replace_line_by_line(file_id, file_content);
            
            % Update caffe stepvalues
            obj.set_caffe_multistep_values();
        end
        
        function set_caffe_multistep_values(~)
            global CONFIG
            
            % Read and get interesting lines (and then also close orig file)
            file_content = fileread(CONFIG.agent_solver_proto);
            
            % Remove all old step values
            file_content = regexprep(file_content, 'stepvalue:\s*\d*', '');

            steps = CONFIG.training_agent_lr_update_steps;
            for i = 1 : numel(steps)
                file_content = strcat(file_content, sprintf('\nstepvalue: %d', steps(i)));
            end
            
            % Replace in train.prototxt
            file_id = fopen(CONFIG.agent_solver_proto, 'w');
            fprintf(file_id, file_content);
            fclose(file_id);
        end
        
        function set_train_proto(obj)

            global CONFIG
            
            % Read and get interesting lines (and then also close orig file)
            file_id = fopen(CONFIG.agent_train_proto);
            [file_content, ~] = obj.read_line_by_line(file_id);

            % Extract the desired batch size to be set
            batch_size = num2str(CONFIG.training_agent_batch_size);
            
            % Change values of desired rows
            line_counter = 1;
            while line_counter <= numel(file_content)
                
                % Extract current line
                line = file_content{line_counter};
                line_counter_start = line_counter;
                
                % 1) Find row with  <<type: "Input">>
                if sum(strfind(line, '"Input"')) > 0 || sum(strfind(line, '"DummyData"')) > 0

                    % 2) Find the closest row with <<shape: >>
                    str_find = strfind(file_content{line_counter}, 'shape');
                    while isempty(str_find) || ~str_find
                        line_counter = line_counter + 1;
                        str_find = strfind(file_content{line_counter}, 'shape');
                    end

                    % 3) Find first, second occurrence of <<dim >>
                    line = file_content{line_counter};
                    idxs_dim_string = strfind(line, 'dim');

                    % 4) Replace all space to next occurrence of <<dim >> with new data
                    idx_start = idxs_dim_string(1); idx_next = idxs_dim_string(2);
                    line_part1 = line(1 : idx_start + 2);
                    line_part2 = line(idx_next : end);
                    new_line = strcat(line_part1, ':', {' '}, batch_size, {' '}, line_part2);
                    new_line = new_line{1};
                    file_content{line_counter} = new_line;
                elseif sum(strfind(line, 'name: "canvas"')) > 0
                    line_counter = line_counter + 3;
                    line = file_content{line_counter};
                    last_dim = strfind(line, 'dim:');
                    last_dim = last_dim(end);
                    if CONFIG.agent_use_angle_canvas
                        canvas_dim = CONFIG.agent_nbr_bins_azim * CONFIG.agent_nbr_bins_elev;
                        line_rest = strcat(num2str(canvas_dim), '} }');
                    else
                        line_rest = '1} }';
                    end
                    new_line = line;
                    new_line = strcat(new_line(1 : last_dim + 4), {' '}, line_rest);
                    file_content{line_counter} = new_line{1};
                elseif sum(strfind(line, 'name: "rig"')) > 0
                    line_counter = line_counter + 3;
                    line = file_content{line_counter};
                    last_dim = strfind(line, 'dim:');
                    last_dim = last_dim(end);
                    if CONFIG.agent_use_rig_canvas
                        canvas_dim = CONFIG.agent_nbr_bins_azim * CONFIG.agent_nbr_bins_elev;
                        line_rest = strcat(num2str(canvas_dim), '} }');
                    else
                        line_rest = '1} }';
                    end
                    new_line = line;
                    new_line = strcat(new_line(1 : last_dim + 4), {' '}, line_rest);
                    file_content{line_counter} = new_line{1};
                elseif sum(strfind(line, 'name: "aux"')) > 0
                    line = file_content{line_counter + 3};
                    last_dim = strfind(line, 'dim:');
                    last_dim = last_dim(end);
                    new_line = strcat(line(1 : last_dim + 4), {' '}, ...
                                num2str(4), '} }');
                    file_content{line_counter + 3} = new_line{1};
                end
                    
                % 2) Make input to fc1 / lstm appropriate based on config
                if sum(strfind(line, 'name: "fc1_input"')) > 0 || sum(strfind(line, 'name: "lstm_input"'))

                    feats = struct('global3d', CONFIG.agent_pose_features_global3d, ...
                                   'camera2d', CONFIG.agent_pose_features_camera2d);

                    use_pred = feats.global3d || feats.camera2d;
                    use_canvas = CONFIG.agent_use_angle_canvas;
                    use_rig_canvas = CONFIG.agent_use_rig_canvas;
                    while ~sum(strfind(line, 'bottom:')) 
                        line_counter = line_counter + 1;
                        line = file_content{line_counter};
                    end
                    end_idx = strfind(line, ':');
                    new_line = strcat(line(1 : end_idx));
                    if use_pred && use_canvas && use_rig_canvas
						new_line = strcat(new_line, ' "data_pred_canvas_rig"');
                    elseif use_pred && use_canvas
                        new_line = strcat(new_line, ' "data_pred_canvas"');
                    elseif use_pred && use_rig_canvas
                        new_line = strcat(new_line, ' "data_pred_rig"');
                    elseif use_pred
                        new_line = strcat(new_line, ' "data_pred"');
                    elseif ~use_pred && use_canvas && use_rig_canvas
                        new_line = strcat(new_line, ' "data_canvas_rig"');
                    elseif ~use_pred && use_canvas
                        new_line = strcat(new_line, ' "data_canvas"');
					elseif ~use_pred && use_rig_canvas
                        new_line = strcat(new_line, ' "data_rig"');
                    else
                        new_line = strcat(new_line, ' "data_flat"');
                    end
                    file_content{line_counter} = new_line;
                end
                     
                % Continue to next line
                line_counter = line_counter_start + 1;
            end
            
            % Set certain things depending on number of modes [REMOVE]
            line_counter = 1;
            while line_counter <= numel(file_content)
                [file_content, ~] = ...
                    obj.replace_non_batch_dim('name: "m"', ...
                                          file_content, line_counter, ...
                                          2);
                [file_content, ~] = ...
                    obj.replace_non_batch_dim('name: "azim_mult"', ...
                                          file_content, line_counter, ...
                                          1);
                [file_content, ~] = ...
                    obj.replace_non_batch_dim('name: "neg_angle_pred"', ...
                                          file_content, line_counter, ...
                                          2);
                [file_content, ~] = ...
                    obj.replace_pred_dims('name: "pred"', file_content, ...
                                          line_counter);
                                    
                 % Continue to next line
                line_counter = line_counter + 1;
            end
            
            % Replace in train.prototxt
            file_id = fopen(CONFIG.agent_train_proto, 'w');
            obj.replace_line_by_line(file_id, file_content);
        end
        
        function [file_content, line_counter] = ...
                    replace_non_batch_dim(~, str, file_content, ...
                                          line_counter, replace_value)
            if sum(strfind(file_content{line_counter}, str))
                replace_value = num2str(replace_value);
                while ~sum(strfind(file_content{line_counter}, 'shape'))
                    line_counter = line_counter + 1;
                end
                line = file_content{line_counter};
                idx_end = strfind(line, '}') - 1;
                idx_end = idx_end(end);
                while ~isstrprop(line(idx_end), 'digit')
                    idx_end = idx_end - 1;
                end
                new_line = line;
                new_line(idx_end) = replace_value;
                file_content{line_counter} = new_line;
            end            
        end
     
        function [file_content, line_counter] = ...
                    replace_pred_dims(obj, str, file_content, line_counter)
            global CONFIG
            if sum(strfind(file_content{line_counter}, str))
                feats = struct('global3d', CONFIG.agent_pose_features_global3d, ...
                               'global3d_cum', CONFIG.agent_pose_features_global3d_cum, ...
                               'camera2d', CONFIG.agent_pose_features_camera2d);
                replace_value = feats.camera2d * 15 * 2 + ... 
                                feats.global3d * 15 * 3 + 1 + ...
                                feats.global3d_cum * 15 * 3;
                if CONFIG.agent_pose_features_global3d_cum
                    replace_value = replace_value + CONFIG.agent_nbr_stacked_cum_preds * 15 * 3;
                end
                replace_value = max(1, replace_value);
                replace_value = num2str(replace_value);
                while ~sum(strfind(file_content{line_counter}, 'input_param'))
                    line_counter = line_counter + 1;
                end
                line = file_content{line_counter};
                idx_end = strfind(line, ':') + 1;
                new_line = line;
                new_line = strcat(new_line(1 : idx_end(end)), {' '}, replace_value, '} }');
                file_content{line_counter} = new_line{1};
            end            
        end

        function [file_content, line_counter] = ...
                    replace_nbr_out(~, str, str_line, file_content, ...
                                    line_counter, replace_value)
            if sum(strfind(file_content{line_counter}, str))
                replace_value = num2str(replace_value);
                while ~sum(strfind(file_content{line_counter}, str_line))
                    line_counter = line_counter + 1;
                end
                line = file_content{line_counter};
                idx_end = strfind(line, ':') + 2;
                new_line = line;
                new_line(idx_end) = replace_value;
                file_content{line_counter} = new_line;
            end            
        end
        
        % Two helpers for read / write file below
        function [file_content, idxs_nonempty_noncommented] = read_line_by_line(obj, file_id)
            file_content = {};
            idxs_nonempty_noncommented = logical([]);
            while 1
                line = fgetl(file_id);
                if ~ischar(line)
                    break;
                end
                space_check = isspace(line);
                if (sum(space_check) == numel(space_check) || strcmp(line(1), '#'))
                    idxs_nonempty_noncommented = [idxs_nonempty_noncommented, false]; %#ok<*AGROW>
                else
                    idxs_nonempty_noncommented = [idxs_nonempty_noncommented, true];
                end
                file_content{end+1} = line; %#ok<*SAGROW>
            end
            fclose(file_id);
        end
        
        function replace_line_by_line(obj, file_id, file_content)
            for i = 1 : numel(file_content)
                if i == numel(file_content)
                    fprintf(file_id,'%s', file_content{i});
                    break
                else
                    fprintf(file_id,'%s\n', file_content{i});
                end
            end
            fclose(file_id);
        end
        
        function nbr_params = count_network_params(obj, net)
            layers_list = net.layer_names;
            % for those layers which have parameters, count them
            nbr_params = 0;
            for j = 1:length(layers_list)
                if ~isempty(net.layers(layers_list{j}).params)
                    feat = net.layers(layers_list{j}).params(1).get_data();
                    nbr_params = nbr_params + numel(feat);
                end
            end
        end
        
        % Hyperdock helper functions
        function tf = in_hyperdock(~)
            tf = exist('/hyperdock/params.json', 'file');
        end
        
        function serie = hyperdock_serie(~, label, x_data, y_data)
           serie = struct('label', label, 'x', x_data, 'y', y_data);
        end
        
        function plot = hyperdock_plot(~, name, x_label, y_label, serie_array)
           plot = struct('name', name, 'x_axis', x_label, 'y_axis', y_label, ...
                         'series', serie_array);
        end

        function json= write_hyperdock_graph(obj, plot_array)
            % Saves a struct of hyperdock to disk. Returns the created
            % json for debugging purposes.
            if obj.in_hyperdock()
                fprintf('Writing Hyperdock graph\n');
                addpath('code/panoptic/json-matlab');
                path = '/hyperdock/graphs.json';
                json = savejson('', plot_array, struct('SingletArray', 1, ...
                                                    'SingletCell', 1));
                if ~strcmp(json(1), '[')
                    json = sprintf('[%s]', json);
                end

                fileID = fopen(path, 'w');
                fprintf(fileID, json);
                fclose(fileID);
            end         
        end
        
        function write_hyperdock_loss(obj, loss, epsiode)
            if obj.in_hyperdock()
                fprintf('Writing Hyperdock loss\n');
                fileID = fopen('last_loss.json', 'w');
                fprintf(fileID,'{"loss": %f, "state": "ok", "ep": %d}', ...
                        loss, epsiode);
                fclose(fileID);
            end
        end
        
        function [R, t, scaling] = get_smpl_matrices(~)
            theta = -pi/2;
            R = [1 0 0; 0 cos(theta) -sin(theta); 0 sin(theta) cos(theta)];
            t = [0 -4.07 0.25];
            scaling = [0.0011 0.0011 0.0012];
        end
        
        function d = iou(~, in1, in2)
            % Much faster than Matlab's built-in version bboxOverlapRatio
            % x1 y1 x2 y2
            %
            % IOU Intersection over union score.
            %   The inputs can be a matrix Nx4 and a 1x4 vector. The output
            %   is a vector with the IOU score of mask2 with each one
            %   of the bounding boxes in mask1
            %
            %   d = iou(in1,in2)
            %
            %   Stavros Tsogkas, <stavros.tsogkas@ecp.fr>
            %   Last update: October 2014
            intersectionBox = [max(in1(:,1), in2(:,1)), max(in1(:,2), in2(:,2)),...
                               min(in1(:,3), in2(:,3)), min(in1(:,4), in2(:,4))];
            iw = intersectionBox(:,3)-intersectionBox(:,1)+1;
            ih = intersectionBox(:,4)-intersectionBox(:,2)+1;
            unionArea = bsxfun(@minus, in1(:,3), in1(:,1)-1) .*...
                        bsxfun(@minus, in1(:,4), in1(:,2)-1) +...
                        bsxfun(@minus, in2(:,3), in2(:,1)-1) .*...
                        bsxfun(@minus, in2(:,4), in2(:,2)-1) - iw.*ih;    
            d = iw .* ih ./ unionArea;
            d(iw <= 0 | ih <= 0) = 0;
        end
        
        function coords = trans_hip_coord(~, pred_coords, annot)
            % Translate hip position from predictor to Panoptic format
            chip_predictor = pred_coords(3, :);
            chip_annot = annot(3, :);
            coords = bsxfun(@plus, pred_coords, chip_annot - chip_predictor);
        end
        
        function hsl = rgb2hsl(~, rgb_in)
            %Converts Red-Green-Blue Color value to Hue-Saturation-Luminance Color value
            %
            %Usage
            %       HSL = rgb2hsl(RGB)
            %
            %   converts RGB, a M [x N] x 3 color matrix with values between 0 and 1
            %   into HSL, a M [x N] X 3 color matrix with values between 0 and 1
            %
            %See also hsl2rgb, rgb2hsv, hsv2rgb
            % (C) Vladimir Bychkovsky, June 2008
            % written using: 
            % - an implementation by Suresh E Joel, April 26,2003
            % - Wikipedia: http://en.wikipedia.org/wiki/HSL_and_HSV
            rgb=reshape(rgb_in, [], 3);
            mx=max(rgb,[],2);%max of the 3 colors
            mn=min(rgb,[],2);%min of the 3 colors
            L=(mx+mn)/2;%luminance is half of max value + min value
            S=zeros(size(L));
            % this set of matrix operations can probably be done as an addition...
            zeroidx= (mx==mn);
            S(zeroidx)=0;
            lowlidx=L <= 0.5;
            calc=(mx-mn)./(mx+mn);
            idx=lowlidx & (~ zeroidx);
            S(idx)=calc(idx);
            hilidx=L > 0.5;
            calc=(mx-mn)./(2-(mx+mn));
            idx=hilidx & (~ zeroidx);
            S(idx)=calc(idx);
            hsv=rgb2hsv(rgb);
            H=hsv(:,1);
            hsl=[H, S, L];
            hsl=round(hsl.*100000)./100000; 
            hsl=reshape(hsl, size(rgb_in));
        end
        
        function rgb = hsl2rgb(~, hsl_in)
            %Converts Hue-Saturation-Luminance Color value to Red-Green-Blue Color value
            %
            %Usage
            %       RGB = hsl2rgb(HSL)
            %
            %   converts HSL, a M [x N] x 3 color matrix with values between 0 and 1
            %   into RGB, a M [x N] X 3 color matrix with values between 0 and 1
            %
            %See also rgb2hsl, rgb2hsv, hsv2rgb
            % (C) Vladimir Bychkovsky, June 2008
            % written using: 
            % - an implementation by Suresh E Joel, April 26,2003
            % - Wikipedia: http://en.wikipedia.org/wiki/HSL_and_HSV
            hsl=reshape(hsl_in, [], 3);
            H=hsl(:,1);
            S=hsl(:,2);
            L=hsl(:,3);
            lowLidx=L < (1/2);
            q=(L .* (1+S) ).*lowLidx + (L+S-(L.*S)).*(~lowLidx);
            p=2*L - q;
            hk=H; % this is already divided by 360
            t=zeros([length(H), 3]); % 1=R, 2=B, 3=G
            t(:,1)=hk+1/3;
            t(:,2)=hk;
            t(:,3)=hk-1/3;
            underidx=t < 0;
            overidx=t > 1;
            t=t+underidx - overidx;

            range1=t < (1/6);
            range2=(t >= (1/6) & t < (1/2));
            range3=(t >= (1/2) & t < (2/3));
            range4= t >= (2/3);
            % replicate matricies (one per color) to make the final expression simpler
            P=repmat(p, [1,3]);
            Q=repmat(q, [1,3]);
            rgb_c= (P + ((Q-P).*6.*t)).*range1 + ...
                    Q.*range2 + ...
                    (P + ((Q-P).*6.*(2/3 - t))).*range3 + ...
                    P.*range4;

            rgb_c=round(rgb_c.*10000)./10000; 
            rgb=reshape(rgb_c, size(hsl_in));
        end
       
        function [recon3D, current_is_garbage] = ...
			get_recon_with_prev(obj, predictions, pose_idxs, idx, visited_cams, nbr_persons, ...
                                nbr_steps, predictions_seq, predictions_seq_ok, ...
                                predictions_seq_all, predictions_seq_ok_all, use_cum, ...
                                intermediate_step, prt)
            
            if ~exist('intermediate_step', 'var')
                intermediate_step = 0;
            end
		    if ~exist('prt', 'var')
                prt = 1;
            end
			
			[~, unique_idxs] = unique(visited_cams(1 : nbr_steps));
            use_indices = true(nbr_steps, 1);
            use_indices(setdiff(1 : nbr_steps, unique_idxs)) = 0;
			preds = predictions{idx};
			p_idxs = pose_idxs(:, idx);
			qualifier = (p_idxs(1 : nbr_steps) ~= -1) .* use_indices;
			current_is_garbage = ~isempty(qualifier) && all(qualifier == 0);
			if current_is_garbage
				% Ensure non-collapse
				qualifier(1) = 1;
			end
			qualifier = [qualifier; false(numel(visited_cams) - numel(qualifier), 1)]; %#ok<*AGROW>
			preds(:, :, ~qualifier) = nan;
			
            if ~use_cum
                recon3D = nanmedian(preds, 3);
                return;
            end
            
            if ~intermediate_step
                preds_seq_ok = predictions_seq_ok;
                preds_seq = predictions_seq;
            else
				preds_seq_ok = predictions_seq_ok_all{nbr_steps};
				preds_seq = predictions_seq_all{nbr_steps};
            end

            if any(cellfun(@isempty, preds_seq))
                recon3D = nanmedian(preds, 3);
            else

                % Get pose reconstruction from previous frame (if any)
				if numel(preds_seq_ok) == nbr_persons
					preds_seq_ok = preds_seq_ok{idx};
					preds_seq = preds_seq{idx};
				else
					preds_seq_ok = [];
					preds_seq = [];
				end
                if ~isempty(preds_seq_ok) && preds_seq_ok(1)
                    pose_recon_prev = preds_seq(:, :, 1);
                else
                    pose_recon_prev = [];
                end

                if current_is_garbage && ~isempty(pose_recon_prev)
                    % Current is garbage, and previous thing exists -- 
                    % just propagate the previous thing
                    recon3D = pose_recon_prev;
                else
                    pred_prev = pose_recon_prev;
                    if ~isempty(pred_prev)
                        pose_recon_prev = obj.trans_hip_coord(pose_recon_prev, preds(:, :, 1));
                        new_pred_with_prev = nanmedian(cat(3, preds, pose_recon_prev), 3);
                    else
                        new_pred_with_prev = nanmedian(preds, 3);
                    end
                    recon3D = new_pred_with_prev;
                end
            end    
        end
        
        function pose_reconstruction = infer_missing_joints(obj, pose_reconstruction)
            nan_idxs = isnan(pose_reconstruction(:, 1));
            mean_nonan = mean(pose_reconstruction(~nan_idxs, :), 1);
            if any(isnan(mean_nonan))
                mean_nonan = zeros(size(mean_nonan));
            end
            pose_reconstruction(nan_idxs, :) = repmat(mean_nonan, nnz(nan_idxs), 1);
        end
    end
end
