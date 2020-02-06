function [X,Y,score] = FindPeaks_release(map, thre)
if(nargin < 2)
    thre = 0.05;
end
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
X(deleIdx,:) = [];
Y(deleIdx,:) = [];
score(deleIdx,:) = [];
end

