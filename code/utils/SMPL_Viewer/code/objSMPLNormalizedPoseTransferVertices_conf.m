function [f, df, scJ_tr] = objSMPLNormalizedPoseTransferVertices_conf(model, pose, betas, kintree_table_old, J_dmhs, extra)

W = (model.confs);
if(isempty(W))
    W = ones(18, 1);
end
% W(W>0.01) = 1;
% W(W<=0.01) = 0;
% keyboard;

gradient = nargout == 2;

[J, dJ_dbetas] = jointRegressor_SMPL(betas, model);
epsilon = 1e-3;
opts.debug = false;
opts.mode = {'double'};
inputs = {castG(pose(:), opts.mode{:}), castG(J, opts.mode{:}), kintree_table_old};

[~, ~, dJ_tr_dJ, results, dr_dp] = poseToJoints_SMPL(inputs{:}, opts);

% regress the vertices
v = poseToVertices_SMPL_deriv(results, J, model, betas);

inds_valid = [5 6 7 2 3 4 9 11 15 16 17 12 13 14 10 8 1];
dmhs = {'Pelvis','RHip','RKnee','RAnkle','LHip','LKnee','LAnkle','Spine1',...
    'Neck','Head','Site','LShoulder','LElbow','LWrist','RShoulder','RElbow','RWrist'};

J_tr = model.regH36MW * v;
cJ_tr = bsxfun(@minus, J_tr, J_tr(17, :));
J_dmhs = J_dmhs(inds_valid, :);

ri(inds_valid) = 1 : length(inds_valid);

kintree_table = ri([0,1,0,4,1,2,2,3,4,5,5,6,0,7,7,8,8,9,9,10,8,11,11,12,12,13,8,14,14,15,15,16]+1);
kintree_table = reshape(kintree_table, 2, [])';

error = 0;
c = size(kintree_table, 1);
for i = 1 : c
    ip = kintree_table(i, 1); %parent
    ic = kintree_table(i, 2); %child
    X = cJ_tr(ic, :) - cJ_tr(ip, :);
    X = X ./ sqrt(sum(X.^2));
    Y = J_dmhs(ic, :) - J_dmhs(ip, :);
    Y = Y ./ sqrt(sum(Y.^2));
    if(ip == 2)
        error = error - min(W(9), W(10)) * X*Y';
    elseif(ip == 5)
        error = error - min(W(12), W(13)) * X*Y';
    elseif(ip == 3)
        error = error - min(W(10), W(11)) * X*Y';
    elseif(ip == 6)
        error = error - min(W(13), W(14)) * X*Y';
    else
        error = error - X*Y';
    end
end
div = 1. / c;
f = error * div;

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

if(nargout == 3)
    scJ_tr = bsxfun(@minus, J_tr(indices, :), (J_tr(indices(3), :) + J_tr(indices(4), :))/2);
    df = [];
end
if(gradient)
    %compute dfdp
    dfdcJ_tr = zeros(size(cJ_tr));
    for i = 1 : c
        ip = kintree_table(i, 1); %parent
        ic = kintree_table(i, 2); %child
        
        X = cJ_tr(ic, :) - cJ_tr(ip, :);
        Y = J_dmhs(ic, :) - J_dmhs(ip, :);
        Y = Y ./ sqrt(sum(Y.^2));
        
        A = sum(X.^2+eps);
        dXndX = [A - X(1).^2 -X(1)*X(2) -X(1)*X(3); -X(2)*X(1)  A - X(2).^2 -X(2)*X(3); -X(3)*X(1) -X(3)*X(2) A - X(3).^2];
        dXndX = dXndX ./ (A.^1.5);
        
        dfdX = - Y * dXndX;
        if(ip == 2)
            dfdX = min(W(9), W(10)) * dfdX;
        end
        if(ip == 5)
            dfdX = min(W(12), W(13)) * dfdX;
        end
        if(ip == 3)
            dfdX = min(W(10), W(11)) * dfdX;
        end
        if(ip == 6)
            dfdX = min(W(13), W(14)) * dfdX;
        end
        dfdcJ_tr(ic, :) = dfdcJ_tr(ic, :) + dfdX;
        dfdcJ_tr(ip, :) = dfdcJ_tr(ip, :) - dfdX;
    end
    
    dfdJ_tr = dfdcJ_tr * div;
    dfdJ_tr(17, :) = dfdJ_tr(17, :) - sum(dfdJ_tr, 1);
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
    df = double(df);
    f = double(f);
%     scJ_tr = double(scJ_tr);
end
end