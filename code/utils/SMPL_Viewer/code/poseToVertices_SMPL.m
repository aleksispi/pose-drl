function [v, J_tr, J_local, J, v2] = poseToVertices_SMPL(pose, betas, kintree_table, model, opts)

[J, ~, v] = jointRegressor_SMPL(betas, model); 
v2 = v;
epsilon = 1e-3;
opts.debug = false;
opts.mode = {'double'};
inputs = {castG(pose(:), opts.mode{:}), castG(J, opts.mode{:}), kintree_table};

[J_tr, ~, ~, results] = poseToJoints_SMPL(inputs{:}, opts);

% keyboard;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
results_new = results;

for i = 1 : size(J, 1)
    results_new(:, 4, i) = results_new(:, 4, i) -  results(:, 1:3, i) * J(i, :)';
end
J_local = squeeze(results_new(:, 4, :))';
% keyboard;

% v_template = model.v_template;
if(isfield(opts, 'v'))
    v = opts.v;
    weights = opts.weights;
else
    weights = model.weights';
end
% keyboard;
% weights = sparse(weights);
T = reshape(results_new, [], size(J, 1)) * weights;
T = reshape(T, 3, 4, []);

% keyboard;
% rest_shape_h = 
% v = T(:, 1, :) * 

% v = squeeze(T(:, 4, :));
v = bsxfun(@times, squeeze(T(:, 1, :)), v(:, 1)') + ...
    bsxfun(@times, squeeze(T(:, 2, :)), v(:, 2)') + ...
    bsxfun(@times, squeeze(T(:, 3, :)), v(:, 3)') + ...
    squeeze(T(:, 4, :));
v = v';
% keyboard;
end