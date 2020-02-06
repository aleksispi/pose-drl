
function [ SMPLr ] = initModel(m, w_reg_main, w_reg_mass, gmm)
SMPLr = [];
SMPLr.bias = m.regressor * m.v_template;
SMPLr.beta_regressor = reshape(m.regressor * reshape(m.shapedirs, 6890, []), size(m.regressor, 1), 3, 10);
SMPLr.weights = m.weights.x;
SMPLr.v_template = m.v_template;
SMPLr.w_reg_beta = 0.005;
SMPLr.w_reg_theta = 0.0005;
SMPLr.w_reg_main = w_reg_main;
SMPLr.w_reg_mass = w_reg_mass;
SMPLr.w_reg_collision = 0.005;
SMPLr.shapedirs = m.shapedirs;
SMPLr.gmm = gmm;
SMPLr.spheres = m.spheres;
SMPLr.vgrouping = m.Vgrouping;
SMPLr.clabels = m.C_semantics;
SMPLr.faces = m.faces;
SMPLr.use_gpu = false;
end
