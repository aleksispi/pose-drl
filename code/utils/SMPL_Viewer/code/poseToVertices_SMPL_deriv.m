function [v_out, dLdresults, dLdJ, dLdb] = poseToVertices_SMPL_deriv(results, J, model, beta, dLdv)

dLdresults = [];
dLdJ = [];

gradient = nargout > 1 && nargin >= 5;

results_new = results;

for i = 1 : size(J, 1)
    results_new(:, 4, i) = results(:, 4, i) -  results(:, 1:3, i) * J(i, :)';
end

weights = model.weights';
vs = model.v_template + reshape(reshape(model.shapedirs, [], size(beta, 1)) * beta, size(model.v_template));

T = reshape(results_new, [], size(J, 1)) * weights;
T = reshape(T, 3, 4, []);

v = bsxfun(@times, squeeze(T(:, 1, :)), vs(:, 1)') + ...
    bsxfun(@times, squeeze(T(:, 2, :)), vs(:, 2)') + ...
    bsxfun(@times, squeeze(T(:, 3, :)), vs(:, 3)') + ...
    squeeze(T(:, 4, :));
v = v';
v_out = v;


if(gradient)
    
    dLdv = dLdv';
    
    dLdT = zeros(size(T));
    
    
    dLdT(:, 1, :) = bsxfun(@times, dLdv, vs(:, 1)');
    dLdT(:, 2, :) = bsxfun(@times, dLdv, vs(:, 2)');
    dLdT(:, 3, :) = bsxfun(@times, dLdv, vs(:, 3)');
    dLdT(:, 4, :) = dLdv;
    
%     keyboard;
    dLdT = reshape(dLdT, [], size(dLdT, 3));
    
    dLdresults = dLdT * weights';
    dLdresults = reshape(dLdresults, size(results));
    
    dLdJ = zeros(size(J));
%     keyboard;
    for i = 1 : size(J, 1)
        dLdresults(:, 1:3, i) = dLdresults(:, 1:3, i) - dLdresults(:, 4, i) * J(i, :);
        dLdJ(i, :) = - results(:, 1:3, i)' * dLdresults(:, 4, i); 
    end
    
    dLdvs = zeros(size(vs'));
    dLdvs(1, :) = sum(bsxfun(@times, squeeze(T(:, 1, :)), dLdv(1, :)), 1);
    dLdvs(2, :) = sum(bsxfun(@times, squeeze(T(:, 2, :)), dLdv(2, :)), 1);
    dLdvs(3, :) = sum(bsxfun(@times, squeeze(T(:, 3, :)), dLdv(3, :)), 1);
    dLdvs = dLdvs';
    dLdb = dLdvs(:)' * reshape(model.shapedirs, [], size(beta, 1));
    dLdb = dLdb';
end

% keyboard;
end