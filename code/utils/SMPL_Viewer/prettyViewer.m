function [out] = prettyViewer(path_code, J_dmhs_orig, opts)

max_fun_evals = 100;

% data loading
gmm = load('code/SMPL/smpl_theta_gmm');
regH36M = load('code/SMPL/regH36M');
regH36MW = sparse(regH36M.W);

if(opts.ismale)
    m = load('code/SMPL/model_male.mat');
else
    m = load('code/SMPL/model_female.mat');
end

SMPLr = initModel(m, [], [], gmm);
SMPLr.confs = [];
SMPLr.J_dmhs = J_dmhs_orig/1000;
SMPLr.pwr = 0.5;
SMPLr.regH36MW = regH36MW;
SMPLr.confs = [];
SMPLr.pwr = 0.5;

kintree_table = m.kintree_table';
%%initial values
pose = zeros(3, 24);
betas = zeros(10, 1);

%% ANGLES optimization
SMPLr.w_reg_theta = 0.0075;
SMPLr.w_reg_beta = 0.01;
SMPLr.w_reg_collision = 0.000;

funObjT = @(x, mask) objSMPLNormalizedPoseTransferVertices_conf(SMPLr, x(1:numel(pose)), x(1+numel(pose):end),kintree_table, SMPLr.J_dmhs, mask);
options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton', 'GradObj', 'on', 'MaxFunEvals', max_fun_evals, 'MaxIter', 200, 'TolFun', 1e-5);
x0 = [pose(:); betas];
maskA = ones(numel(pose) + numel(betas), 1);
maskA(end-9:end) = 0;
objMaskedPose = @(x) funObjT(x, maskA);
[xstar, ~] = fminunc(objMaskedPose, x0, options);
pose_transf_corr = xstar(1:numel(pose));
betas_transf_corr = xstar(1+numel(pose):end);
pose_transf = pose_transf_corr;
betas_transf = betas_transf_corr;

%% RMSE optimization
SMPLr.w_reg_theta = 0.001;
SMPLr.w_reg_beta = 0.01;
SMPLr.w_reg_collision = 0.001;

funObjR = @(x, mask) objSMPLPoseBetasTransferVertices(SMPLr, x(1:numel(pose)), x(1+numel(pose):end), kintree_table, SMPLr.J_dmhs, mask);
x0 = [pose_transf(:); betas_transf];
maskA = [ones(numel(pose), 1); [1 1 1 1 1 1 1 1 1 1]'];
objMaskedEuc = @(x) funObjR(x, maskA);
[xstar, ~] = fminunc(objMaskedEuc, x0, options);
[~,~,~,fEuc] = objMaskedEuc(xstar);

%% Get final results and render
pose_transf_final = xstar(1:numel(pose));
betas_transf_final = xstar(1+numel(pose):end);

direction = 'front';
color = opts.color;

if(opts.ismale)
    [I, out, h] = renderSMPL([path_code 'SMPL/model_male.mat'], pose_transf_final, betas_transf_final, opts.T, color, direction);
else
    [I, out, h] = renderSMPL([path_code 'SMPL/model_female.mat'], pose_transf_final, betas_transf_final, opts.T, color, direction);
end

% Apply material after each render
material([0.2 0.9 0.2 5 0.2]);

if numel(findall(gcf,'Type','light')) < 1
    % Only add one light per image
    camlight;
end

end