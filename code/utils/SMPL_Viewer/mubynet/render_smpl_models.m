figure(4);
xstar_t = reshape(xstar_translation(1:end-1), np, 3);
colors = parula(np);
ind = ones(np, 1);
for id_p = 1: np
  
    pose_pred = pose_transf_final(id_p,:);
    betas_pred = betas_transf_final(id_p,:);

    [v_new_pred, J_new] = poseToVertices_SMPL(pose_pred', betas_pred', kintree_table, SMPLr);
    [predicted_w_hip, predicted] = getSMPLvIndicesFromDMHS(SMPLr.faces, double(v_new_pred), J_new, SMPLr.regH36MW);

    T_smpl = predicted_w_hip(1,:);

     x_final = xstar_t(id_p, :);
  
    T_final = x_final - T_smpl/1000;
    
     if   ind(id_p)
         renderSMPL_GenerateData('code/SMPL/model_male.mat', pose_pred', betas_pred', T_final, colors(id_p,:) )
     else

          renderSMPL_GenerateData('code/SMPL/model_female.mat', pose_pred', betas_pred', T_final, colors(id_p,:) )
    end
    material([0.2 0.9 0.2 5 0.2]);
    if id_p==np
        camlight;

    end

    hold on;
end