function [I, out, h] = renderSMPL(path_to_model_m, pose_new, betas_new, T_new, color, direction)

% Notes: 
% bopts, not given!
% viz not given!
bopts = struct;
bopts.scale_vp = 1;
bopts.debug = true;
bopts.lighting = 'gouraud';
bopts.C = [];
bopts.pId = 1;
viz = 2;
bopts.color = color;

if strcmp(direction, 'front')
    bopts.cpos = [0 0 -1];
elseif strcmp(direction, 'back')
    bopts.cpos = [0 0 1];
elseif strcmp(direction, 'left')
    bopts.cpos = [1 0 0];
elseif strcmp(direction, 'right')
    bopts.cpos = [-1 0 0];
else
    error('Invalid position');
end

m = load(path_to_model_m);


SMPLr = [];
SMPLr.bias = m.regressor * m.v_template;
SMPLr.beta_regressor = reshape(m.regressor * reshape(m.shapedirs, 6890, []), size(m.regressor, 1), 3, 10);
SMPLr.weights = m.weights.x;

SMPLr.v_template = m.v_template;
SMPLr.shapedirs = m.shapedirs;

kintree_table = m.kintree_table';

[v_new_m, J_new] = poseToVertices_SMPL(pose_new, betas_new, kintree_table, SMPLr);


v_new = v_new_m * 0.5 + v_new_m * 0.5;

bopts.ctar = [0 0 0];

ar = 1920/1080;

set(gcf, 'Renderer', 'opengl');
set(gcf,'Color',[1,1,1]);
axis equal;
camproj('perspective');
camva(50);
set(gca, 'DataAspectRatio', [1, 1, 1]);
ax = gca;               % get the current axis
ax.Clipping = 'off';    % turn clipping off
axis off;
campos(bopts.cpos)
camtarget(bopts.ctar);
camup([0 -1 0]);
grid on;

colors = distinguishable_colors(size(m.regressor, 1)+1);


[~, cindex] = max(SMPLr.weights, [], 2);

C = repmat(bopts.color, [6890,1]);

positions = bsxfun(@plus, v_new, T_new);
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
    ymean = sum(triangles(:, :, 2), 2)/3;
    f = find(ymean >= ymax + (ymin - ymax) * bopts.dt);
end

faces = m.faces(f, :)+1;

visible = sum(bsxfun(@times, positions, [0 0 -1]), 2);

h = patch('FaceAlpha', 1, 'FaceLighting', bopts.lighting, 'Faces', faces, ...
    'Vertices', positions, 'FaceVertexCData', C,  'FaceColor','interp', 'EdgeColor', 'none', 'LineWidth', 0.01, 'EdgeLighting', bopts.lighting, 'Marker', '*', 'MarkerSize', 10, 'MarkerEdgeColor', 'none');

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