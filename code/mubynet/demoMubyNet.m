function [J3D] = demoMubyNet(img_path, output_name)

% demoMubyNet - apply MubyNet CNN model on a single image
% [J3D] = demoMubyNet(img_path, output_name)
%
% OUTPUT :
% J3D              - array containing the corresponding 3D pose
%                    predictions for the determined number of 
%                    persons (npx17x3)

%
% INPUT  :
% img_path         - path to image used for testing
%                  - it should contain a single person inside a bounding box
% output_name      - output name for mat file with results corresponding to
%                    image to img_path
%                  - the mat file will be saved by default in ./data/results/

%% default arg values

if(nargin == 0)
    img_path = './data/images/im0054.jpg';
end
if(nargin < 2)
    output_name = 'results_im0054';
end

%% path setup

close all;
addpath('code');
try
    caffe.reset_all();
catch
end
%% load image

img = imread(img_path);
img = imresize(img, [300, NaN]);
%% 2d/3d/grouping network

mode = 1;
param = config_release(mode, 230000, 1);
net_sampling = caffe.Net(param.model.deployFile, param.model.caffemodel, 'test');
mode = 2;
param = config_release(mode, 230000, 1);
net_scoring = caffe.Net(param.model.deployFile, param.model.caffemodel, 'test');
%% sampling and grouping

[final_score] = applyModel_3d(img, param, net_sampling, 1);
%% BIP decoding

[sample_scores, candidates, subset, limbs3d, limbs2d] = BIP_decode(final_score, param, net_scoring);
np = size(limbs3d, 1);
for k = 1 : np
    J3D = squeeze(limbs3d(k, :, :))';
end

%% draw 2d

limbSeq = [2 3; 2 6; 3 4; 4 5; 6 7; 7 8; 2 9; 9 10; 10 11; 2 12; 12 13; 13 14; 2 1; 1 15; 15 17; 1 16; 16 18; 3 17; 6 18];
colors = {'r', 'g', 'b', 'c', 'm', 'y', 'k'};
figure,
imshow(img);
hold on;
for i = 1:np
    for l = 1:size(limbSeq, 1)
        if(limbs2d(i, 3, limbSeq(l,1)) * limbs2d(i, 3, limbSeq(l,2)))
            plot([limbs2d(i, 1, limbSeq(l,1)), limbs2d(i, 1, limbSeq(l,2))], [limbs2d(i, 2, limbSeq(l,1)), limbs2d(i, 2, limbSeq(l,2))], colors{i});
        end
    end
end
    
%% non-max suppression and draw

figure,
imshow(img);
hold on;
colors = {'*r', '*g', '*b', '*c', '*m', '*y', '*k', '^r', '^g', '^b', '^c', '^m', '^y', '^k', 'or', 'og', 'ob', 'oc', 'om', 'oy', 'ok'};
for i = 1:18
    [X,Y,score] = FindPeaks_release(final_score(:,:,2+i), param.thre1);
    for j = 1:length(X)
        text(Y(j)-10, X(j), param.model.part_str{i});
        plot(Y(j), X(j), colors{i});
    end
end

%% translation inference 

[h, w, ~] = size(img);
cx = w/2;
cy = h/2;
l2d = limbs2d(:, :, [3, 6, 4, 7, 5, 8, 9, 12, 10, 13, 11, 14]);
l2d = cat(3, (limbs2d(:, :, 7) + limbs2d(:, :, 8)) * 0.5, l2d);
l3d = limbs3d(:, :, [1, 15, 12, 16, 13, 17, 14, 2, 5, 3, 6, 4, 7]);
loss_all = @(x) loss_proj(reshape(x(1:end-1), [], 3), x(end), permute(limbs3d(:, :, [15, 12, 16, 13, 17, 14, 2, 5, 3, 6, 4, 7]), [1 3 2]), permute(limbs2d(:, :, [3, 6, 4, 7, 5, 8, 9, 12, 10, 13, 11, 14]), [1 3 2]), cx, cy);
x0 = repmat([0, 0, 3.1], [np 1]);
x0 = x0(:);
x0(end+1) = 60 * pi / 180;

lb = -Inf(size(x0));
lb(end) = 15*pi/180;
ub = Inf(size(x0));
ub(end) = 120*pi/180;
options = optimoptions('fmincon','Display','final','Algorithm','interior-point', 'MaxFunEvals', Inf, 'MaxIter', 500);
xstar = fmincon(loss_all,x0,[],[],[],[],lb,ub, [], options);

[~, ~, stats] = loss_all(xstar);
T = squeeze(stats.x3d(:, 1, :));

%% draw 3d

figure, hold on;
for k = 1 : np
    prob = squeeze(limbs2d(k, 3, :));
    if(mean(prob(prob>0)) < 0.5 || sum(prob > 0) <= 8)
        continue;
    end
    J3D = squeeze(limbs3d(k, :, :))';
    plotSkel(J3D*1000 + repmat(T(k, :), [17 1]) * 1000, 'b*');
    axis image
    axis equal
end

%% save results

save(['./data/results/' output_name '.mat'], 'J3D', 'limbs2d');
