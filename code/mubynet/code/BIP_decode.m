function [scores_sampling, candidates, subset, limbs3d, limbs2d] = BIP_decode(heatMaps, param, net_sampling)
% vis = 1;
%figure;

limbs3d = [];
limbs2d = [];

count = 0;
candidates = [];

maximum = cell(18, 1);
for j = 1:18
    [Y,X,score] = findPeaks(heatMaps(:,:,2+j), param.thre1);
    temp = (1:numel(Y)) + count;
    maximum{j} = [X, Y, score, reshape(temp,[numel(Y),1])];
    candidates = [candidates; X Y score ones([numel(Y),1])*j];
    count = count + numel(Y);
end

limbSeq = [2 3; 2 6; 3 4; 4 5; 6 7; 7 8; 2 9; 9 10; 10 11; 2 12; 12 13; 13 14; 2 1; 1 15; 15 17; 1 16; 16 18; 3 17; 6 18];
% the middle joints heatmap correpondence
% last number in each row is the total parts number of that person
% the second last number in each row is the score of the overall configuration
subset = [];
% keyboard
contor = 0;
S = [];

cand_all = [];
idx_all = [];
for k = 1:18
    cand = maximum{k};
    idx = ones(size(cand,1),1)*k;
    idx_all = [idx_all; idx];
    cand_all = [cand_all;cand];
    
end
start_j = zeros(length(idx_all), 1);
end_j = zeros(length(idx_all),1);

% keyboard;

%%%%%%%%%%  COMPUTE LIMBS SAMPLING LOCATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% construct_sampling_time = tic;

mid_num = 10;

max_candidates = max(cell2mat(cellfun(@size, maximum, 'UniformOutput', 0)));
max_candidates = max_candidates(1);
feature_samples = zeros(max_candidates * max_candidates * size(limbSeq, 1), 128, mid_num);
contor_fs = 0;
for k = 1: size(limbSeq, 1)
    candA = maximum{limbSeq(k,1)};
    candB = maximum{limbSeq(k,2)};
    nA = size(candA,1);
    nB = size(candB,1);
    for i = 1:nA
        for j = 1:nB
            
            contor_fs = contor_fs+1;
            x = round(linspace(candA(i,1),candB(j,1), mid_num));
            y = round(linspace(candA(i,2),candB(j,2), mid_num));
            
            
            for lm = 1:mid_num
                feature_samples(contor_fs, :, lm) = heatMaps(y(lm), x(lm), 198:end);
            end
            
            
            contor = contor+1;
            id = find(idx_all==limbSeq(k,1));
            start_j(contor) = id(1)+i-1;
            id = find(idx_all==limbSeq(k,2));
            end_j(contor)=id(1)+j-1;
        end
    end
end
if(contor_fs <= 3)
    scores_sampling = [];
    return;
end
feature_samples = feature_samples(1:contor_fs, :, :);
% fprintf(1, 'The time for construct_sampling_time is %f\n', toc(construct_sampling_time));

%%%%%%%%%%%%% SAMPLE LOCATIONS %%%%%%%%%%%%%%%
% sample_time = tic;
feature_samples_per = permute(feature_samples, [3 2 1]);
net_sampling.blob_vec(1).reshape(size(feature_samples_per))
net_sampling.reshape()
input_data = {single(feature_samples_per)};
net_sampling.forward(input_data);
scores_sampling = net_sampling.blobs(['scores']).get_data();
scores_sampling = scores_sampling(2, :);
% fprintf(1, 'The time for sampling is %f\n', toc(sample_time));


%%%%%%%%%%% CONSTRUCT CONSTRAINTS MATRICES %%%%%%%%%%%%%%%%%%%%%
% constraints_time = tic;

S = double(scores_sampling);
S(S <= 0.1) = -100000;

A= zeros(1,size(S,2));
c = 0;
ind = 1;
for k = 1: size(limbSeq, 1)
    candA = maximum{limbSeq(k,1)};
    candB = maximum{limbSeq(k,2)};
    nA = size(candA,1);
    nB = size(candB,1);
    
    for i = 1: nA
        c = c+1;
        A(c,ind:ind+nB-1) = 1;
        ind = ind+nB;
    end
end


% A= zeros(32,size(S,2));
c = c+1;
ind = 0;
for k = 1: size(limbSeq, 1)
    candA = maximum{limbSeq(k,1)};
    candB = maximum{limbSeq(k,2)};
    nA = size(candA,1);
    nB = size(candB,1);
    
    %     for i = 1: nA
    for j = 1:nB
        
        %             A(c,ind+((j-1)*nB)+j:nB:nB*nA+ind) = 1;
        A(c,ind+j:nB:nB*nA+ind) = 1;
        c = c+1;
    end
    
    
    %     end
    ind = ind+nA*nB;
end
% fprintf(1, 'The time for constraints is %f\n', toc(constraints_time));
%% BIP
options = optimoptions('intlinprog', 'Display', 'off');
f = -S;
intcon = [1: length(S)];
b = ones(1,size(A,1));
ub = ones(1,length(S));
lb = zeros(1,length(S));

x = intlinprog(f, intcon, A,b, [], [],lb,ub, options);





%% Decoding

% keyboard;
%%
selected_limbs = find(x==1);
skeletons = {};

while true
    if(isempty(selected_limbs))
        break;
    end
    skeletons{end+1} = [];
    skeletons{end}(end+1) = selected_limbs(1);
    selected_limbs(1) = [];
    
    append = true;
    while append == true
        append = false;
        N = length(skeletons{end});
        for k = 1 : N
            i = skeletons{end}(k);
            j_down = find(start_j(i) == end_j(selected_limbs));
            
            if(~isempty(j_down))
                j_down = j_down(1);
                skeletons{end}(end+1) = selected_limbs(j_down);
                selected_limbs(j_down) = [];
                append = true;
            end
            
            j_up = find(end_j(i) == start_j(selected_limbs));
            if(~isempty(j_up))
                j_up = j_up(1);
                skeletons{end}(end+1) = selected_limbs(j_up);
                selected_limbs(j_up) = [];
                append = true;
            end
            
            %%%%%%%%%%%%% special cases (more connections at same level)
            
            j_down = find(start_j(i) == start_j(selected_limbs));
            
            if(~isempty(j_down))
                j_down = j_down(1);
                skeletons{end}(end+1) = selected_limbs(j_down);
                selected_limbs(j_down) = [];
                append = true;
            end
            
            j_up = find(end_j(i) == end_j(selected_limbs));
            if(~isempty(j_up))
                j_up = j_up(1);
                skeletons{end}(end+1) = selected_limbs(j_up);
                selected_limbs(j_up) = [];
                append = true;
            end
        end
    end
end


%%
limbs2d = zeros(length(skeletons), 3, 18);
limbs3d = zeros(length(skeletons), 3*17);
for id = 1 : length(skeletons)
    if(length(skeletons{id}) < 3)
        continue;
    end
    l_start = start_j(skeletons{id});
    l_end =  end_j(skeletons{id});
    selected = union(l_start, l_end);
    jt = candidates(selected, 4);
    limbs2d(id, :, jt) = candidates(selected, 1:3)';
    
    ct = 0;
    for l = 1:length(l_start)
        index = [l_start(l) l_end(l)];
        X = candidates(index,1);
        Y = candidates(index,2);
        prob = mean(candidates(index, 3));
        mp = round([X(1) Y(1)]*0.5 + [X(2) Y(2)]*0.5);
        limbs3d(id, 4:end) = limbs3d(id, 4:end) + prob * reshape(heatMaps(mp(2), mp(1), 22:69), [1 48]);
        ct = ct + prob;
    end
    limbs3d(id, 4:end) = limbs3d(id, 4:end) / ct;
end
limbs3d = reshape(limbs3d, [length(skeletons) 3 17]);

limbs2dnew = [];
limbs3dnew = [];
cnt = 0;
for k = 1: size(limbs2d,1)
    prob = squeeze(limbs2d(k, 3, :));
    if(mean(prob(prob>0)) < -1 || sum(prob > 0) < 5) %%%%%%%%%%% changed from 5
    else
        cnt = cnt+1;
        limbs2dnew(cnt, :,:) = limbs2d(k,:,:);
        limbs3dnew(cnt, :,:) = limbs3d(k,:,:);
    end
end
limbs2d = limbs2dnew;
limbs3d = limbs3dnew;

end
function [X,Y,score] = findPeaks(map, thre)
%filter = fspecial('gaussian', [3 3], 2);
%map_smooth = conv2(map, filter, 'same');

map_smooth = map;
map_smooth(map_smooth < thre) = 0;

map_aug = -1*zeros(size(map_smooth,1)+2, size(map_smooth,2)+2);
map_aug1 = map_aug;
map_aug2 = map_aug;
map_aug3 = map_aug;
map_aug4 = map_aug;

map_aug(2:end-1, 2:end-1) = map_smooth;
map_aug1(2:end-1, 1:end-2) = map_smooth;
map_aug2(2:end-1, 3:end) = map_smooth;
map_aug3(1:end-2, 2:end-1) = map_smooth;
map_aug4(3:end, 2:end-1) = map_smooth;

peakMap = (map_aug > map_aug1) & (map_aug > map_aug2) & (map_aug > map_aug3) & (map_aug > map_aug4);
peakMap = peakMap(2:end-1, 2:end-1);
[X,Y] = find(peakMap);
score = zeros(length(X),1);
for i = 1:length(X)
    score(i) = map(X(i),Y(i));
end

if isempty(X)
    return;
end

deleIdx = [];
flag = ones(1, length(X));
for i = 1:length(X)
    if(flag(i)>0)
        for j = (i+1):length(X)
            if norm([X(i)-X(j),Y(i)-Y(j)]) <= 6
                flag(j) = 0;
                deleIdx = [deleIdx;j];
                
            end
        end
    end
end
% if ~isempty(deleIdx)
%     keyboard
% end
X(deleIdx,:) = [];
Y(deleIdx,:) = [];
score(deleIdx,:) = [];
end
