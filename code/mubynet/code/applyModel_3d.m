function heatMaps = applyModel_3d(test_image, param, net, rescale)

%% select model and other parameters from variable param
model = param.model;
boxsize = model.boxsize;
%%

oriImg = test_image;
% use given scale
multiplier = param.scale_search; %(0.7:0.3:1.3)*scale0; %(0.5:0.3:1.4)*scale0;

pad = cell(1, length(multiplier));
ori_size = cell(1, length(multiplier));

score = cell(1, length(multiplier));
for m = 1:length(multiplier)
    
    scale = multiplier(m);
    imageToTest = imresize(oriImg, scale);
    ori_size{m} = size(imageToTest);
    
    bbox = [boxsize, max(ori_size{m}(2),boxsize)];
    if(rescale)
        [imageToTest, pad{m}] = padHeight(imageToTest, model.padValue, bbox);
    end
    imageToTest = preprocess(imageToTest, 0.5);
    if numel(imageToTest) > 3728000*3
        disp('Image Size Too Large!');
        continue;
    end
    [h, w, ~] = size(imageToTest);
    [geomX, geomY] = ind2sub([h w], 1 : h*w);
    geomX = reshape(geomX - 1, [h w]) / max(geomX(:) - 1);
    geomY = reshape(geomY - 1, [h w]) / max(geomY(:) - 1);
    geomX = imresize(geomX, 0.125, 'nearest');
    geomY = imresize(geomY, 0.125, 'nearest');
    [h, w, ~] = size(geomY);
    [geomX, geomY] = ind2sub([h w], 1 : h*w);
    geomX = reshape(geomX - 1, [h w]) / max(geomX(:) - 1);
    geomY = reshape(geomY - 1, [h w]) / max(geomY(:) - 1);
    geomXY = cat(3, geomX, geomY);
    
    
    if(length(net.inputs) >= 2)
        if(strcmp(net.inputs{2}, 'geomXY'))
            net.blob_vec(2).reshape([size(geomXY) 1]);
        end
    end
    net.blob_vec(1).reshape([size(imageToTest) 1])
    net.reshape()
    %     keyboard;
    [score{m}] = applyDNNLS(imageToTest, net, geomXY);
    
    pool_time = size(imageToTest,1) / size(score{m},1);
    if(rescale)
        score{m} = imresize(score{m}, pool_time);
        score{m} = resizeIntoScaledImg(score{m}, pad{m});
    else
        score{m} = imresize(score{m}, pool_time);
    end
end
for m = 2 : length(multiplier)
    score{m} = imresize(score{m}, [size(score{1}, 1) size(score{1}, 2)], 'bilinear');
end
final_score = zeros(size(score{1}));
for m = 1 : length(score)
    final_score = final_score + score{m};
end
final_score = final_score / length(score);
% final_score = imresize(final_score, [size(oriImg, 2) size(oriImg, 1)], 'bilinear');
heatMaps = permute(final_score, [2 1 3]);
end
function img_out = preprocess(img, mean)
img_out = double(img)/255;
img_out = double(img_out) - mean;
img_out = permute(img_out, [2 1 3]);

if size(img_out,3) == 1
    img_out(:,:,3) = img_out(:,:,1);
    img_out(:,:,2) = img_out(:,:,1);
end

img_out = img_out(:,:,[3 2 1]); % BGR for opencv training in caffe !!!!!
end
function [scores] = applyDNN(images, net)
input_data = {single(images)};
net.forward(input_data);
try
    L2 = net.blobs(['final_L2_scaled']).get_data();
catch
    L2 = net.blobs(['Mconv7_stage6_L2']).get_data();
end
L3 = net.blobs(['final_map']).get_data();
scores = cat(3, L2(:,:,1:end-1), L3);
end

function [scores] = applyDNNLS(images, net, geomXY)
input_data = {single(images)};
input_data{end+1} = geomXY;
net.forward(input_data);
try
    L1 = net.blobs(['conv2_FM']).get_data();
    L2 = net.blobs(['concat_stage1_FM']).get_data();
    scores = cat(3, L2, L1);
catch
    scores = net.blobs(['concat_stage1_FM']).get_data();
end
end

function [img_padded, pad] = padHeight(img, padValue, bbox)
h = size(img, 1);
w = size(img, 2);
h = min(bbox(1),h);
bbox(1) = ceil(bbox(1)/8)*8;
bbox(2) = max(bbox(2), w);
bbox(2) = ceil(bbox(2)/8)*8;
pad(1) = floor((bbox(1)-h)/2); % up
pad(2) = floor((bbox(2)-w)/2); % left
pad(3) = bbox(1)-h-pad(1); % down
pad(4) = bbox(2)-w-pad(2); % right

img_padded = img;
pad_up = repmat(img_padded(1,:,:), [pad(1) 1 1])*0 + padValue;
img_padded = [pad_up; img_padded];
pad_left = repmat(img_padded(:,1,:), [1 pad(2) 1])*0 + padValue;
img_padded = [pad_left img_padded];
pad_down = repmat(img_padded(end,:,:), [pad(3) 1 1])*0 + padValue;
img_padded = [img_padded; pad_down];
pad_right = repmat(img_padded(:,end,:), [1 pad(4) 1])*0 + padValue;
img_padded = [img_padded pad_right];
%cropping if needed
end
function score = resizeIntoScaledImg(score, pad)
np = size(score,3)-1;
score = permute(score, [2 1 3]);
if(pad(1) < 0)
    padup = cat(3, zeros(-pad(1), size(score,2), np), ones(-pad(1), size(score,2), 1));
    score = [padup; score]; % pad up
elseif(pad(1) > 0)
    score(1:pad(1),:,:) = []; % crop up
end

if(pad(2) < 0)
    padleft = cat(3, zeros(size(score,1), -pad(2), np), ones(size(score,1), -pad(2), 1));
    score = [padleft score]; % pad left
elseif(pad(2) > 0)
    %     score(:,1:pad(2),:) = []; % crop left
    index = true(1, size(score, 2));
    index(1:pad(2)) = false;
    score = score(:,index, :);
end

if(pad(3) < 0)
    paddown = cat(3, zeros(-pad(3), size(score,2), np), ones(-pad(3), size(score,2), 1));
    score = [score; paddown]; % pad down
elseif(pad(3) > 0)
    score(end-pad(3)+1:end, :, :) = []; % crop down
end

if(pad(4) < 0)
    padright = cat(3, zeros(size(score,1), -pad(4), np), ones(size(score,1), -pad(4), 1));
    score = [score padright]; % pad right
elseif(pad(4) > 0)
    %score(:,end-pad(4)+1:end, :) = []; % crop right
    
    index = true(1, size(score, 2));
    index(end-pad(4)+1:end) = false;
    score = score(:,index, :);
end
score = permute(score, [2 1 3]);
end
