function [I, out, h] = renderSMPL_GenerateData(path_to_model_m, pose_new, betas_new, T_new, color, bopts, viz )

if(~exist('bopts', 'var'))
    bopts.scale_vp = 1;
    bopts.debug = true;
    bopts.lighting = 'gouraud';
    bopts.C = [];
    bopts.pId = 1;
    viz = 2;
    bopts.color = color;
end


m = load(path_to_model_m);

SMPLr = [];
SMPLr.bias = m.regressor * m.v_template;
SMPLr.beta_regressor = reshape(m.regressor * reshape(m.shapedirs, 6890, []), size(m.regressor, 1), 3, 10);
SMPLr.weights = m.weights.x;
% SMPLr.weights = bsxfun(@times, m.regressor, 1./sum(m.regressor, 1))';
SMPLr.v_template = m.v_template;
SMPLr.shapedirs = m.shapedirs;

kintree_table = m.kintree_table';

[v_new_m, J_new] = poseToVertices_SMPL(pose_new, betas_new, kintree_table, SMPLr);


v_new = v_new_m * 0.5 + v_new_m * 0.5;
% v_new(1)
%0.0812
if(isfield(bopts, 'v2d'))
    v_new(:, 1:2) = bopts.v2d;
end
if(isfield(bopts, 'vnew'))
    v_new = bopts.vnew;
end


if(~isfield(bopts, 'cpos'))
    bopts.cpos = [0 0 -1];
    %     bopts.cpos = [-0.5 0 -1];
    bopts.ctar = [0 0 0];
    %      bopts.ctar = [-0.5 0 0];
end
ar = 1920/1080;
% if(bopts.pId == 1)
%  figure('Visible', 'On', 'units','normalized','position',[0. 0. bopts.scale_vp bopts.scale_vp])
% end
% figure('Visible', 'On', 'units','normalized','position',[0. 0. 1 0.898])
%  figure('Visible', 'On', 'units','normalized','position',[0. 0. 1 1])
%  figure('Visible', 'On', 'units','normalized','position',[0.1 0. 0.898,1]);
% figure('Visible', 'On', 'units','normalized','position',[0. 0. 1  1]);

% set(gcf,'RendererMode','manual');
set(gcf, 'Renderer', 'opengl');
set(gcf,'Color',[1,1,1]);
% set(gcf,'PaperPositionMode', 'auto');
% clf;
% set(gcf, 'units', 'normalized');
% set(gcf, 'outerposition', [0 0 1 1]);
% set(gcf,'Color',[1.0 1.0 1.0]);
% axis equal;
if ~viz
    axis([-4 4 -4 4 -4 4]);
    axis off;
    campos([0, 0, 0])
    camtarget([0, 0, 1]);
    camorbit(0, 30, 'camera');
    camzoom(4.5);
elseif viz == 2
    if(bopts.pId == 1)
        axis equal;
        %         axis manual;
        camproj('perspective');
    end
    camva(50);
    set(gca, 'DataAspectRatio', [1, 1, 1]);
    ax = gca;               % get the current axis
    ax.Clipping = 'off';    % turn clipping off
    %     keyboard;
    axis off;
    campos(bopts.cpos)
    camtarget(bopts.ctar);
    camup([0 -1 0]);
    %     view(0, -58);
else
    axis off;
end
% camzoom
% R = T_new(3)/2;
% alpha = -pi/5;
%  campos([0, R*sin(alpha), R*cos(alpha)])
%  v = [0,0,R] - [0, R*sin(alpha), R*cos(alpha)];
%  v = v/norm(v);
% camtarget(v);

% camva(90);
% camup([0 -1 0]);
% R = vrrotvec2mat([1, 0, 0, pi/5]);
% v_new = bsxfun(@plus, bsxfun(@minus, v_new, [0 0 2])*R', [0 0 2]);
% camlight;
% if(~strcmp(bopts.lighting, 'flat'))
%     light('Position',[0 0 1]),
%     light('Position',[0 0 -1]),
% end
grid on;
% hold on;
% rotate3d on;

colors = distinguishable_colors(size(m.regressor, 1)+1);
% [~, cindex] = max(SMPLr.weights, [], 2);

% colors(:) = 0;
% colors(14, :) = [1 0 0];
if(bopts.debug == 1)
    [~, cindex] = max(SMPLr.weights, [], 2);
    if(~isfield(bopts, 'C') || isempty(bopts.C))
        C = colors(cindex, :);
    else
        C = bopts.C;
    end
    % C = colors(cindex, :);
    %C = m.C_semantics;
    %     keyboard;
    %     C = SMPLr.weights * colors;
    C = repmat(bopts.color, [6890,1]);
    if(strcmp(bopts.lighting, 'flat'))
        
        patch('FaceAlpha', 1, 'FaceLighting', bopts.lighting, 'Faces', m.faces+1, ...
            'Vertices', bsxfun(@plus, v_new, T_new), 'FaceVertexCData',C,  'FaceColor','interp', 'EdgeColor', 'none', 'LineWidth', 0.01, 'EdgeLighting', bopts.lighting);
    else
        positions = bsxfun(@plus, v_new, T_new);
        %         keyboard;
        if(isfield(bopts, 'Rc'))
            positions = (bopts.Rc * bsxfun(@minus, positions, bopts.Tc)')';
            positions = bsxfun(@plus, positions, bopts.ctar);
        end
        %         positions(:, 1) = positions(:, 1) * 0.76 ;
        cdata = bsxfun(@plus, v_new(:, 3), T_new(:, 3));
        if(isfield(bopts, 'min'))
            zdata = bsxfun(@plus, v_new(:, 3), T_new(:, 3));
            cdata = repmat(bopts.color, size(zdata, 1), 1) .* repmat((zdata - bopts.min) / (bopts.max - bopts.min), 1, 3);
        end
        
        ymax = max(positions(:, 2));
        ymin = min(positions(:, 2));
        f = 1 : length(m.faces);
        if(isfield(bopts, 'dt'))
            triangles = positions(m.faces+1, :);
            triangles = reshape(triangles, [], 3, 3);
            %             keyboard;
            ymean = sum(triangles(:, :, 2), 2)/3;
            f = find(ymean >= ymax + (ymin - ymax) * bopts.dt);
        end
        %         bopts.lighting
        faces = m.faces(f, :)+1;
        %         v_normals = STLVertexNormals(m.faces(f, :)+1, positions);
        visible = sum(bsxfun(@times, positions, [0 0 -1]), 2);
        %         face_i = visible<0.0 & positions(:, 2) <= 0.4844 & positions(:, 2) >= 0.2777;
        %         save('/home/andreiz/face_vertices.mat', 'face_i');
        %         hold on;
        %         scatter3(positions(face_i, 1), positions(face_i, 2), positions(face_i, 3)+0.05, 7, 'blue');
        %         C(face_i, :) = repmat(colors(end, :), [sum(face_i) 1]);
        h = patch('FaceAlpha', 1, 'FaceLighting', bopts.lighting, 'Faces', faces, ...
            'Vertices', positions, 'FaceVertexCData', C,  'FaceColor','interp', 'EdgeColor', 'none', 'LineWidth', 0.01, 'EdgeLighting', bopts.lighting, 'Marker', '*', 'MarkerSize', 10, 'MarkerEdgeColor', 'none');
        %          hold on;
        %          plot3(positions(bopts.ids(:, 1), 1), positions(bopts.ids(:, 1), 2), positions(bopts.ids(:, 1), 3), 'r*');
        %          plot3(positions(bopts.ids(:, 2), 1), positions(bopts.ids(:, 2), 2), positions(bopts.ids(:, 2), 3), 'g*');
        %          bopts.points = bsxfun(@plus, bopts.points, T_new);
        %          plot3(bopts.points(:, 1), bopts.points(:, 2), bopts.points(:, 3), 'b*');
        %          plot3(positions(bopts.ids(:, 3), 1), positions(bopts.ids(:, 3), 2), positions(bopts.ids(:, 3), 3), 'b*');
        %          h = patch('FaceAlpha', 1, 'FaceLighting', bopts.lighting, 'Faces', faces, ...
        %              'Vertices', positions, 'FaceVertexCData', C,  'FaceColor', 'interp', 'EdgeColor', 'none', 'LineWidth', 0.01, 'EdgeLighting', bopts.lighting, 'Marker', '*', 'MarkerSize', 10, 'MarkerEdgeColor', 'none');
        %          patch('FaceLighting','gouraud', 'Faces', m.faces+1, 'Vertices', bsxfun(@plus, v_new, T_new), 'FaceColor','red', 'EdgeColor', 'none');
        %          camlight('headlight');
        %% compute normals
        %         keyboard;
        
        
        
        %         keyboard;
        %         hold on;
        %         for i = 1:24
        %             f = find(cindex == i);
        %             v = mean(positions(f, :), 1);
        %             text(v(1), v(2), v(3) + 0.3, sprintf('%d', i), 'Color', 'magenta');
        %         end
        %         hold on;
        %         handr = getHandVertices(positions, 0);
        %         handl = getHandVertices(positions, 1);
        %         scatter3(handr(:, 1), handr(:, 2), handr(:, 3), 'bo');
        %         scatter3(handl(:, 1), handl(:, 2), handl(:, 3), 'ro');
        %         joints = bsxfun(@plus, J_new, T_new);
        %         scatter3(joints(:, 1), joints(:, 2), joints(:, 3), 'yo');
        %         keyboard;
        %         rotate(h, [1 0 0], 70, [0, 0, 0]);
    end
    %     bias =  0.3;
    %     scatter3(J_new(:, 1), J_new(:, 2), J_new(:, 3) + bias, 'go');
    %     for i = 1 : size(J_new, 1)
    %         text(J_new(i, 1), J_new(i, 2), J_new(i, 3) + bias, sprintf('%d', i), 'Color', 'blue');
    %     end
elseif(bopts.debug == 0)
    patch('FaceLighting','gouraud', 'Faces', m.faces+1, 'Vertices', bsxfun(@plus, v_new, T_new), 'FaceColor','red', 'EdgeColor', 'none');
elseif(bopts.debug == 2)
    [xs,ys,zs] = sphere;
    pts_s(:, 1) = xs(:);
    pts_s(:, 2) = ys(:);
    pts_s(:, 3) = zs(:);
    spheres = bopts.spheres;
    [new_centers, ~] = poseToVertices_SMPL(pose_new, betas_new, kintree_table, bopts.model, struct('v', spheres.centers, 'weights', spheres.weights'));
    
    for i = 1 : numel(spheres.id)
        
        pts_new = bsxfun(@plus, pts_s * spheres.radius(i), new_centers(i, :));
        pts_new = bsxfun(@plus, pts_new, T_new);
        C = pts_new(:, 3);
        x = reshape(pts_new(:, 1), [21 21]);
        y = reshape(pts_new(:, 2), [21 21]);
        z = reshape(pts_new(:, 3), [21 21]);
        
        fvc = surf2patch(x,y,z,'triangles');
        %         patch(fvc, 'FaceAlpha', 1, 'FaceLighting', bopts.lighting, 'EdgeLighting', bopts.lighting, 'FaceColor','none', 'EdgeColor', 'green');
        patch(fvc, 'FaceAlpha', 0.7, 'FaceLighting', 'gouraud', 'FaceVertexCData', C,  'FaceColor','interp', 'EdgeColor', 'none', 'LineWidth', 0.01, 'EdgeLighting', 'gouraud');
    end
end


if(nargout > 0)
    out.SMPLr = SMPLr;
    out.v_new = v_new;
    out.J_new = J_new;
    % keyboard;
    fig = gcf;
    fig.InvertHardcopy = 'off';
    %     if(~strcmp(version, '8.3.0.532 (R2014a)'))
    set(fig, 'PaperPositionMode', 'auto');
    I = print(fig,'-RGBImage', '-opengl', '-r0');
    %     else
    %         I = [];
    %     end
end

% I = getimage(gca);

end