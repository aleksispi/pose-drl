function [J, dJ, v_shaped] = jointRegressor_SMPL(beta, m)


if(~exist('m','var') || isempty(m))
    m = load('code/SMPL/model.mat');
end

gradient = nargout > 1;
% v = m.v_template;
% shapedirs = m.shapedirs;
% regressor = m.regressor;
% 
% Nv = size(v, 1);
% v_shaped = v + reshape(reshape(shapedirs, [], size(beta, 1)) * beta, size(v));
% 
% %v_posed = v_shaped + posedirs.dot(posemap(bs_type)(pose))
% 
% v = v_shaped;
% 
% J = regressor * v_shaped;

J = m.bias + reshape(reshape(m.beta_regressor, [], size(beta, 1)) * beta, size(m.bias));

if(nargout == 3)
    v = m.v_template;
    shapedirs = m.shapedirs;
    v_shaped = v + reshape(reshape(shapedirs, [], size(beta, 1)) * beta, size(v));
end
if(gradient)
    dJ = permute(m.beta_regressor, [3 1 2]);
    dJ = reshape(dJ, numel(beta), []);
end
end