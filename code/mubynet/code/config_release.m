function param = config_release(id, caffepath)
%% set this part

global CONFIG

if(nargin < 3)
    caffepath = '/windows/phd/rl_pose/weak-rl-pose/caffe/matlab';
end
GPUdeviceNumber = CONFIG.gpu_id;
param.use_gpu = 2;

if id == 1 % limb sampling
    param.scale_search = [1];
    param.thre1 = 0.1;
    param.thre2 = 0.05; 
    param.thre3 = 0.5; 
    param.thre5 = 0;
    param.model.caffemodel = CONFIG.muby_weights;
    param.model.deployFile = CONFIG.muby_deploy_proto_sampling;
    param.model.description = 'COCO - limb sampling trained on 3D, 2D and geom';
    param.model.boxsize = 368;
    param.model.padValue = 128;
    param.model.np = 18; 
    param.model.part_str = {'nose', 'neck', 'Rsho', 'Relb', 'Rwri', ... 
                             'Lsho', 'Lelb', 'Lwri', ...
                             'Rhip', 'Rkne', 'Rank', ...
                             'Lhip', 'Lkne', 'Lank', ...
                             'Leye', 'Reye', 'Lear', 'Rear', 'pt19'};
elseif id == 2 % limb scoring
    param.scale_search = [1];
    param.thre1 = 0.1;
    param.thre2 = 0.05; 
    param.thre3 = 0.5; 
    param.thre5 = 0;
    param.model.caffemodel = CONFIG.muby_weights;
    param.model.deployFile = CONFIG.muby_deploy_proto_scoring;
    param.model.description = 'COCO - limb scoring trained on 3D, 2D and geom';
    param.model.boxsize = 368;
    param.model.padValue = 128;
    param.model.np = 18; 
    param.model.part_str = {'nose', 'neck', 'Rsho', 'Relb', 'Rwri', ... 
                             'Lsho', 'Lelb', 'Lwri', ...
                             'Rhip', 'Rkne', 'Rank', ...
                             'Lhip', 'Lkne', 'Lank', ...
                             'Leye', 'Reye', 'Lear', 'Rear', 'pt19'};
end
disp(caffepath);
addpath(caffepath);
caffe.set_mode_gpu();
caffe.set_device(GPUdeviceNumber);
end