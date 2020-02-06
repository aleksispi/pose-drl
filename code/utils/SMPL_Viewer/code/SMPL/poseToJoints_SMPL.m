function [J_tr, dJdp, dJdJ, results, dres_dP] = poseToJoints_SMPL(pose, J, kintree_table, opts)
% this is the equivalent of lbs.global_rigid_transformation function
if(~exist('opts', 'var'))
    opts.debug = true;
    opts.mode = {'single'};
end
if(opts.debug == true)
    smpl_corres = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]+1;
    links = [9 8; 8 7; 9 10; 10 11; 11 12; 9 3; 3 2; 2 1; 10 4; 4 5; 5 6];
end
if(exist('pose', 'var') && isempty(pose))
    m = load('code/SMPL/model.mat');
    pose = reshape(m.example_pose, 3, []);
end
if(~exist('kintree_table', 'var') || isempty(kintree_table))
    m = load('code/SMPL/model.mat');
    kintree_table = m.kintree_table';
end

nelems = size(kintree_table, 1);
pose = reshape(pose, 3, []);
jacobian = nargout > 1;
% keyboard;

if(~jacobian) % just forward
    results{1} = [rodrigues(pose(:, 1)) J(1, :)'];
    
    for i = 2 : nelems
        parent = kintree_table(i, 1)+1;
        localM = [rodrigues(pose(:, i)) (J(i, :) - J(parent, :))'];
        localM = [localM; 0.0 0.0 0.0 1.0];
        results{i} = results{parent} * localM;
    end
    J_tr = cellfun(@(x) x(:, 4), results, 'UniformOutput', false);
    J_tr = [J_tr{:}]';
    dJdp = [];
    
    % keyboard;
    if(opts.debug == true)
        J_v = J_tr(smpl_corres, :);
        Xn = J_v;
        for i = 1 : size(links, 1)
            j1 = links(i,1);
            j2 = links(i,2);
            line([Xn(j1, 1) Xn(j2, 1)], [Xn(j1, 2) Xn(j2, 2)], [Xn(j1, 3) Xn(j2, 3)],'LineWidth',4,'Color',[.8 .0 .8]);
        end
        axis equal;
    end
    
elseif(jacobian) %we are asked for derivatives
  
    pose = reshape(pose, 3, []);
    dM_dP = zeros(3, 4*4, opts.mode{:});
    results = zeros(3, 4, size(kintree_table, 1), opts.mode{:});
    % pose X 3*4 X J
    dres_dP = zeros(numel(pose), 3*4, size(J, 1), opts.mode{:});
    idTriple = @(i) [3*i-2:3*i];
    
   
    [R, dR] = rodrigues(pose(:, 1), opts.mode);
    results(:, :, 1) = [R J(1, :)'];
    
    dres_dP(idTriple(1), 1:9, 1) = dR;
    
    if(nargout >= 3)
        dJdJ = zeros(3*size(J, 1), 3*size(J, 1));
        dJdJ(idTriple(1), idTriple(1)) = eye(3);
    end
    
    for i = 2 : size(kintree_table, 1)
%         keyboard;
        parent = kintree_table(i, 1)+1;
        [R, dR] = rodrigues(pose(:, i), opts.mode);
        limb = (J(i, :) - J(parent, :))';
        M = [R limb; 0 0 0 1];
        results(:, :, i) = results(:, :, parent) * M;
        
        
        % compute dJ_tr/dJ
       
        if(nargout >= 3)
           
            dJdJ(:, idTriple(i)) = dJdJ(:, idTriple(parent));
            dJdJ(idTriple(i), idTriple(i)) = dJdJ(idTriple(i), idTriple(i)) + squeeze(results(1:3, 1:3, parent))';
            dJdJ(idTriple(parent), idTriple(i)) = dJdJ(idTriple(parent), idTriple(i)) -squeeze(results(1:3, 1:3, parent))';
        end
        
        % compute dJ_tr/dP
       
        dM_dP(:, [1 2 3 5 6 7 9 10 11]) = dR;
        
        dres_dP_parent = reshape(dres_dP(:, :, parent), [], 4);
        
        sz = size(dres_dP(:, :, i));
        right_t = reshape(dM_dP, 3, 4, 4);
        right_t = permute(right_t, [2 3 1]);
        right_t = reshape(right_t, 4, []);
        right_t = results(:, :, parent) * right_t;
        right_t = reshape(right_t, [3 4 3]);
        right_t = reshape(right_t, 12, 3)';
        
        left = reshape(dres_dP_parent * M,  3*nelems, 12);
        left(idTriple(i), :) = left(idTriple(i), :) + right_t;
        
        %         keyboard;
        dres_dP(:, :, i) = reshape(left, sz);
        %sum(dJdJ(:))
    end
   
    J_tr = squeeze(results(:, 4, :))';
    dres_dP = permute(dres_dP, [3 1 2]);
    dres_dP = reshape(dres_dP, nelems, 3*nelems, 3, 4);
    dJdP = dres_dP(:, :, :, 4);
    dJdp = permute(dJdP, [3 1 2]);
    %pose x joints
    dJdp = reshape(dJdp, 3*nelems, [])';
%      keyboard;
end
% keyboard;
end