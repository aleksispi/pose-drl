function [f, df, scJ_tr, fEuc] = objSMPLPoseBetasTransferVertices(model, pose, betas, kintree_table, J_dmhs, extra)

gradient = nargout > 1;

% regress joints based on beta also
if(gradient)
    [J, dJ_dbetas] = jointRegressor_SMPL(betas, model);
else
    J = jointRegressor_SMPL(betas, model); 
end

epsilon = 1e-3;
opts.debug = false;
opts.mode = {'double'};
inputs = {castG(pose(:), opts.mode{:}), castG(J, opts.mode{:}), kintree_table};



[~, ~, dJ_tr_dJ, results, dr_dp] = poseToJoints_SMPL(inputs{:}, opts);

% regress the vertices
v = poseToVertices_SMPL_deriv(results, J, model, betas);

inds_valid = [5 6 7 2 3 4 9 11 15 16 17 12 13 14 10 8 1];

J_tr = model.regH36MW * v;
cJ_tr = bsxfun(@minus, J_tr, J_tr(17, :));
J_dmhs = J_dmhs(inds_valid, :);


pwr = model.pwr;
% save intermediate values
resJ = cJ_tr - J_dmhs;
fs = power(sum( resJ .^ 2, 2) + epsilon, pwr);
ifs = 2 * pwr ./ (fs .^ ((1-pwr)/pwr));
div = 1 ./ size(fs, 1);

% compute the loss function
f = sum(fs) * div;
fEuc = f;
% add beta penalty
if(model.w_reg_beta ~= 0)
    f = f + model.w_reg_beta * sum(betas .^ 2);
end

% add pose prior penalty
if(model.w_reg_theta ~= 0)
    [f_gmm_theta, df_gmm_theta] = GMMThetaPrior(pose, model.gmm);
    f = f + model.w_reg_theta * f_gmm_theta;
end

% add collision penalty
if(model.w_reg_collision ~= 0)
    [f_collision, ~, df_coll_theta] = poseToVertices_SMPL_eff(results, J, model, dr_dp);
    f = f + model.w_reg_collision * f_collision;
end


if(~gradient)
%     figure,
%     plot3(s*cJ_tr(:, 1), s*cJ_tr(:, 2), s*cJ_tr(:, 3), 'b*'); hold on;
%     plot3(J_dmhs(:, 1), J_dmhs(:, 2), J_dmhs(:, 3), 'r*'); hold on;
%     view(0, 90);
%     axis('equal');
end
if(nargout >= 3)
%     invs_inds = 1 : length(inds_valid);
    for i = 1 : length(inds_valid)
        fcor = find(inds_valid == i);
        scJ_tr(i, :) = cJ_tr(fcor, :);
    end
%     scJ_tr = cJ_tr;
    df = [];
end
if(gradient)
    
    dfdcJ_tr = div * bsxfun(@times, resJ, ifs);
    dfdJ_tr = dfdcJ_tr;
    dfdJ_tr(17, :) = dfdJ_tr(17, :) - sum(dfdJ_tr, 1);
%     keyboard
    dfdv = model.regH36MW' * dfdJ_tr;
    
    [~, dfdresults, dfdJd, dfdbd] = poseToVertices_SMPL_deriv(results, J, model, betas, dfdv);
    % compute df/dtheta
    dr_dp = permute(dr_dp, [3 4 1 2]);
    dr_dp = reshape(dr_dp, [], size(dr_dp, 4));
    dfdtheta = dfdresults(:)' * dr_dp;
    dfdtheta = dfdtheta';
    
    dJ_tr_dJ = dJ_tr_dJ';
    dfdresults = squeeze(dfdresults(:, 4, :));
    
    dfdJ = reshape(dfdresults, 1, []) * dJ_tr_dJ;
    dfdJ = dfdJd + reshape(dfdJ, 3, 24)';
    
    % compute df/dbeta
    dfdb = dfdbd + dJ_dbetas * dfdJ(:);
    
    if(model.w_reg_theta ~= 0)
        dfdtheta = dfdtheta + model.w_reg_theta * df_gmm_theta;
    end
    if(model.w_reg_collision ~= 0)
        dfdtheta = dfdtheta + model.w_reg_collision * df_coll_theta;
    end
    if(model.w_reg_beta ~= 0)
        dfdb = dfdb + model.w_reg_beta * 2 * betas;
    end
    
    df(numel(pose) + 1 : numel(pose) + numel(betas)) = dfdb';
    df(1 : numel(pose)) = dfdtheta';
%     keyboard;
    if(exist('extra', 'var'))
        df = df .* reshape(extra, size(df));
    end
end
end