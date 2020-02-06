classdef CameraRig < handle
    % Represents layout etcetera of the camera rig of a panoptic scene

    properties
        camera_calibs
        nbr_cameras
        radius
        center
        elev_angle_max
        elev_angle_min
        elev_angle_mean
        elev_angle_span_half
        cam_coords
        princ_axes
    end

    methods
        
        function obj = CameraRig(camera_calibs)
            % Setup camera rig settings
            obj.camera_calibs = camera_calibs;
            obj.nbr_cameras = numel(obj.camera_calibs);
            obj.setup_cam_rig()
        end
        
        function setup_cam_rig(obj)
            obj.update_rig(obj.camera_calibs)
            obj.sphere_fit(obj.cam_coords(:, [1, 3, 2]));
        end
        
        function update_rig(obj, new_camera_calibs)
            obj.nbr_cameras = numel(new_camera_calibs);
            obj.princ_axes = nan(obj.nbr_cameras, 3);
            obj.cam_coords = nan(obj.nbr_cameras, 3);
            for i = 1 : obj.nbr_cameras
                calib = new_camera_calibs{i};
                P = calib.P;
                princ_ax = P(3, 1 : 3);
                obj.princ_axes(i, :) = 50 * princ_ax / norm(princ_ax);
                cam_ctr = null(P);
                cam_ctr = cam_ctr / cam_ctr(end);
                obj.cam_coords(i, :) = cam_ctr(1 : 3)';
            end
            
        end
        
        function show_cam_rig(obj, new_fig, show_all_cams, ...
                              best_cam_ord, show_top_prct)
            if ~exist('show_all_cams', 'var'); show_all_cams = 1; end
            if ~exist('best_cam_ord', 'var'); best_cam_ord = nan; end
            if ~exist('show_top_prct', 'var'); show_top_prct = 1.0; end
            if new_fig; figure; hold on; end
        
            linewidth = 5;
            markersize = 10;
            
            if show_all_cams
                if isnan(best_cam_ord)
                    plot3(obj.cam_coords(:, 1), obj.cam_coords(:, 2), obj.cam_coords(:, 3), 'ro', 'MarkerSize', markersize, 'LineWidth', linewidth);
                    quiver3(obj.cam_coords(:, 1), obj.cam_coords(:, 2), obj.cam_coords(:, 3), ...
                            obj.princ_axes(:, 1), obj.princ_axes(:, 2), obj.princ_axes(:, 3), 'r', 'LineWidth', 2.0, 'AutoScaleFactor', 0.3, 'ShowArrowHead', 'off');
                else
                    reds = linspace(0, 255, round(size(obj.cam_coords, 1) * show_top_prct))';
                    blues = flipud(reds);
                    color_range_cams = flipud([reds, zeros(size(reds)), blues]) / 255;
                    for i = 1 : round(size(obj.cam_coords, 1) * show_top_prct)
                        curr_color = color_range_cams(i, :);
                        j = best_cam_ord(i);
                        plot3(obj.cam_coords(j, 1), obj.cam_coords(j, 2), obj.cam_coords(j, 3), 'o', ...
                              'MarkerEdgeColor', curr_color, 'MarkerFaceColor', curr_color);
                        quiver3(obj.cam_coords(j, 1), obj.cam_coords(j, 2), obj.cam_coords(j, 3), ...
                                obj.princ_axes(j, 1), obj.princ_axes(j, 2), obj.princ_axes(j, 3), 'g');

                    end
                end
            end
            obj.plot_sphere();
            axis equal;
        end
        
        function fig = show_cam_choices(obj, camera_indices, fig)
            if ~exist('fig', 'var')
                fig = figure('visible', 'off'); hold on;
            end
                
            linewidth = 5.0;
            markersize = 10;
            quiver_scale = 1.3;
            
            c = camera_indices;
            latest = c(end);
            old = c(1:end-1);
            
            % Plot the latest camera
            
            plot3(obj.cam_coords(latest, 1), obj.cam_coords(latest, 2), obj.cam_coords(latest, 3), 'bo', 'MarkerSize', markersize, 'LineWidth', linewidth);
            quiver3(obj.cam_coords(latest, 1), obj.cam_coords(latest, 2), obj.cam_coords(latest, 3), ...
                    obj.princ_axes(latest, 1) * quiver_scale, obj.princ_axes(latest, 2) * quiver_scale, obj.princ_axes(latest, 3) * quiver_scale, 'b', 'LineWidth', linewidth,'AutoScale', 'off', 'ShowArrowHead', 'off');
                
            if numel(old) > 0
                plot3(obj.cam_coords(old, 1), obj.cam_coords(old, 2), obj.cam_coords(old, 3), 'ro', 'MarkerSize', markersize, 'LineWidth', linewidth);
                quiver3(obj.cam_coords(old, 1), obj.cam_coords(old, 2), obj.cam_coords(old, 3), ...
                        obj.princ_axes(old, 1) * quiver_scale, obj.princ_axes(old, 2) * quiver_scale, obj.princ_axes(old, 3) * quiver_scale, 'r', 'LineWidth', linewidth, 'AutoScale', 'off', 'ShowArrowHead', 'off');
            end
                
            obj.plot_sphere();
            camtarget([1.5687 -206.8446 1.0687]);
            campos(1.0e+03 * [0.0610 -3.6110 -4.0565]);
            axis off;
        end
        
        function show_cam_rig_smpl(obj, transformation, new_fig, show_all_cams, ...
                              best_cam_ord, show_top_prct)
            if ~exist('show_all_cams', 'var'); show_all_cams = 1; end
            if ~exist('best_cam_ord', 'var'); best_cam_ord = nan; end
            if ~exist('show_top_prct', 'var'); show_top_prct = 1.0; end
            if new_fig; figure; hold on; end
            
            global CONFIG

            linewidth = 2.5;
            markersize = 10;
            
            cam_calibs = transformation.cam_calibs;
            scaling = transformation.scaling;
            scaling = scaling(:, [1, 3, 2]) / CONFIG.panoptic_scaling_factor;
            
            for i = 1 : numel(cam_calibs)
                cam_calibs{i}.t = cam_calibs{i}.t .* scaling';
                cam_calibs{i}.P = cam_calibs{i}.K * [cam_calibs{i}.R, cam_calibs{i}.t];
            end
            
            % save old
            old_calibs = obj.camera_calibs;
            obj.camera_calibs = cam_calibs;
            obj.setup_cam_rig();
            
            if show_all_cams
                if isnan(best_cam_ord)
                    plot3(obj.cam_coords(:, 1), obj.cam_coords(:, 2), obj.cam_coords(:, 3), 'bo', 'MarkerSize', markersize, 'LineWidth', linewidth);
                    quiver3(obj.cam_coords(:, 1), obj.cam_coords(:, 2), obj.cam_coords(:, 3), ...
                            obj.princ_axes(:, 1), obj.princ_axes(:, 2), obj.princ_axes(:, 3), 'b', 'LineWidth', 2.0, 'AutoScaleFactor', 0.3, 'ShowArrowHead', 'off');
                else
                    reds = linspace(0, 255, round(size(obj.cam_coords, 1) * show_top_prct))';
                    blues = flipud(reds);
                    color_range_cams = flipud([reds, zeros(size(reds)), blues]) / 255;
                    for i = 1 : round(size(obj.cam_coords, 1) * show_top_prct)
                        curr_color = color_range_cams(i, :);
                        j = best_cam_ord(i);
                        plot3(obj.cam_coords(j, 1), obj.cam_coords(j, 2), obj.cam_coords(j, 3), 'o', ...
                              'MarkerEdgeColor', curr_color, 'MarkerFaceColor', curr_color);
                        quiver3(obj.cam_coords(j, 1), obj.cam_coords(j, 2), obj.cam_coords(j, 3), ...
                                obj.princ_axes(j, 1), obj.princ_axes(j, 2), obj.princ_axes(j, 3), 'g');

                    end
                end
            end
            
            obj.plot_sphere();
            axis equal;
            
            % Go back to original
            obj.camera_calibs = old_calibs;
            obj.setup_cam_rig();
        end
        
        function sphere_fit(obj, X)
            
            % Fit center (origin) and radius of sphere to data
            A=[mean(X(:,1).*(X(:,1)-mean(X(:,1)))), ...
                2*mean(X(:,1).*(X(:,2)-mean(X(:,2)))), ...
                2*mean(X(:,1).*(X(:,3)-mean(X(:,3)))); ...
                0, ...
                mean(X(:,2).*(X(:,2)-mean(X(:,2)))), ...
                2*mean(X(:,2).*(X(:,3)-mean(X(:,3)))); ...
                0, ...
                0, ...
                mean(X(:,3).*(X(:,3)-mean(X(:,3))))];
            A=A+A.';
            B=[mean((X(:,1).^2+X(:,2).^2+X(:,3).^2).*(X(:,1)-mean(X(:,1))));...
                mean((X(:,1).^2+X(:,2).^2+X(:,3).^2).*(X(:,2)-mean(X(:,2))));...
                mean((X(:,1).^2+X(:,2).^2+X(:,3).^2).*(X(:,3)-mean(X(:,3))))];
            obj.center = (A\B).';
            obj.radius = sqrt(mean(sum([X(:,1)-obj.center(1),X(:,2) ...
                                - obj.center(2),X(:,3)-obj.center(3)].^2,2)));
            obj.center = obj.center([1, 3, 2]);
            
            % Once the center (origin) and radius of the sphere have been
            % found, we can compute the min and max elevation angles
            
            % The first step in doing so is finding the vectors from the
            % center pointing to the camera with max and min y-coords,
            % respectively (we take minus because rig is "upside-down")
            [~, cam_y_max_idx] = max(obj.cam_coords(:, 2));
            [~, cam_y_min_idx] = min(obj.cam_coords(:, 2));
            angles_cam_y_max = obj.global_angles_cam(cam_y_max_idx);
            angles_cam_y_min = obj.global_angles_cam(cam_y_min_idx);
            obj.elev_angle_min = angles_cam_y_min(2);
            obj.elev_angle_max = angles_cam_y_max(2);
            obj.elev_angle_mean = 0.5 * (obj.elev_angle_max + obj.elev_angle_min);
            obj.elev_angle_span_half = 0.5 * (obj.elev_angle_max - obj.elev_angle_min);
        end
        
        function plot_sphere(obj)
            thetas = linspace(0, 2 * pi, 40);
            %phis = linspace(0.43*pi, pi, 20); UNCOMMENT if you want full sphere
            phis = linspace(obj.elev_angle_min, obj.elev_angle_max, 20);
            [Phis, Thetas] = meshgrid(phis, thetas);
            r = obj.radius;
            c = obj.center;
            X = r * sin(Phis) .* cos(Thetas) + c(1);
            % Set minus, since upside down
            Y = -r * cos(Phis) + c(2);
            Z = r * sin(Phis) .* sin(Thetas) + c(3);
            surf(X, Y, Z, 'FaceAlpha', 0.1); axis equal; hold on;
        end
        
        function global_angles = global_angles_cam(obj, camera_idx)
       
            % Extract camera coordinates
            cam = obj.cam_coords(camera_idx, :);
            
            % Compute global azimuth angle
            global_angle_azim = angle(cam(1) + 1i * cam(3));

            % Next we'll compute the global elevation angle
            
            % The first step in doing so is finding the vector from the
            % center pointing to the camera coordinates
            cam_vec = (cam - obj.center)';
            
            % Next, rotate this cam_vec so that it points
            % in along the x-axis as seen from above
            angle_cam = angle(cam_vec(1) + cam_vec(3) * 1i);
            rot_max_mat = [cos(angle_cam), 0, sin(angle_cam); ...
                           0,               1, 0; ...
                           -sin(angle_cam), 0, cos(angle_cam)];
            cam_vec = rot_max_mat * cam_vec;
            
            % Next compute the max-min-elevation angle (we also need below
            % transformation due to upside-down y-axis etc)
            global_angle_elev = pi / 2 + angle(cam_vec(1) + cam_vec(2) * 1i);
                
            % Finally we can return the global angles
            global_angles = [global_angle_azim, global_angle_elev];
        end
    end
end