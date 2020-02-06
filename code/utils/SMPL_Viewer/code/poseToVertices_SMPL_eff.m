function [L, v_out, dLdtheta] = poseToVertices_SMPL_eff(results, J, model, dresults_dtheta)

dLdtheta = [];

gradient = nargout > 1 && nargin >= 4;

results_new = results;

for i = 1 : size(J, 1)
    results_new(:, 4, i) = results(:, 4, i) -  results(:, 1:3, i) * J(i, :)';
end

weights = model.spheres.weights';
v = model.spheres.centers;


T = reshape(results_new, [], size(J, 1)) * weights;
T = reshape(T, 3, 4, []);

v = bsxfun(@times, squeeze(T(:, 1, :)), v(:, 1)') + ...
    bsxfun(@times, squeeze(T(:, 2, :)), v(:, 2)') + ...
    bsxfun(@times, squeeze(T(:, 3, :)), v(:, 3)') + ...
    squeeze(T(:, 4, :));
v = v';
v_out = v;



[is, js] = find(model.spheres.collisions);

% moving
dist_s = sum((v_out(is, :) - v_out(js, :)).^2, 2);
dist_s_rel = dist_s' ./ (model.spheres.radius(is) + model.spheres.radius(js)).^2;
fc = find(dist_s_rel < 1); % collision
% rest
rdist_s = sum((model.spheres.centers(is, :) - model.spheres.centers(js, :)).^2, 2);
rdist_s_rel = rdist_s' ./ (model.spheres.radius(is) + model.spheres.radius(js)).^2;
alphas = ones(numel(rdist_s), 1) * 0.09;
alphas(rdist_s_rel < 1.4) = 1.4 * 0.09 ./ rdist_s_rel(rdist_s_rel < 1.4);

% keyboard;
if(~isempty(fc))
    errors = exp(- dist_s_rel(fc)'./alphas(fc));
    L = sum(errors);
else
    L = 0;
end
% keyboard;
if(gradient)
    
    if(isempty(fc))
        dLdtheta = zeros(72, 1);
        return;
    end
    v = model.spheres.centers;
    
    dLddt = errors .* (-1./alphas(fc));
    dLdd = zeros(length(dist_s_rel), 1);
    dLdd(fc) = dLddt;
    
    dLdd = (dLdd' ./ (model.spheres.radius(is) + model.spheres.radius(js)).^2);
    
    dLdd = bsxfun(@times, 2*(v_out(is, :) - v_out(js, :)), dLdd');
    dLdv = zeros(size(v));
    
    % can precompute in sparse form!
    C = unique(is);
    dLdv(C, :) = bsxfun(@eq, C, (is(:)).') * dLdd;
    C = unique(js);
    dLdv(C, :) = dLdv(C, :) - bsxfun(@eq, C, (js(:)).') * dLdd;
    
    
    dLdv = dLdv';
    
    dLdT = zeros(size(T));
    
    
    dLdT(:, 1, :) = bsxfun(@times, dLdv, v(:, 1)');
    dLdT(:, 2, :) = bsxfun(@times, dLdv, v(:, 2)');
    dLdT(:, 3, :) = bsxfun(@times, dLdv, v(:, 3)');
    dLdT(:, 4, :) = dLdv;
    
%     keyboard;
    dLdT = reshape(dLdT, [], size(dLdT, 3));
    
    dLdresults = dLdT * weights';
    dLdresults = reshape(dLdresults, size(results));
    
%     keyboard;
    for i = 1 : size(J, 1)
        dLdresults(:, 1:3, i) = dLdresults(:, 1:3, i) - dLdresults(:, 4, i) * J(i, :);
    end
    
    dresults_dtheta = permute(dresults_dtheta, [3 4 1 2]);
    dresults_dtheta = reshape(dresults_dtheta, [], size(dresults_dtheta, 4));
    dLdtheta = dLdresults(:)' * dresults_dtheta;
    dLdtheta = dLdtheta';
end

% keyboard;
end