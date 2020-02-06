

function [D  ] = get_intersection_spheres( v1, v2, model1, model2 )
 v1 = reshape(v1, [60,1,3]);
    v2 = reshape(v2, [1,60,3]);
d = repmat(v1, [1,60,1]) - repmat(v2, [60,1,1]);

r1 = model1.spheres.radius;
r2 =  model2.spheres.radius;

R = (repmat(r1', [1,60]) + repmat(r2, [60,1])).^2;
D = sum(d.^2,3)./R;

% k = 0;
% % dist = zeros(size(v1,1)*size(v2,1),1 );
% for i = 1: size(v1,1)
%  for j = 1: size(v2,1)
%      k = k+1;
%
% %    dist(i,j) = sum((v1(i,:) -v2(j,:)).^2)./(model1.spheres.radius(i) + model2.spheres.radius(j)).^2;
% dist(i,j) = sum((v1(i,:) -v2(j,:)).^2)./(model1(i) + model2(j)).^2;
%  end

% end
% dist_s = sum((v_ou(is, :) - v_out(js, :)).^2, 2);
% dist_s_rel = dist_s' ./ (model.spheres.radius(is) + model.spheres.radius(js)).^2;
% function [dist  ] = get_intersection_spheres( model1, model2 )
% k = 0;
% dist = zeros(length(model1.spheres.id)*length(model2.spheres.id) );
% for i = 1: length(model1.spheres.id)
%  for j = 1: length(model2.spheres.id)
%      k = k+1;
%      keyboard;
%     dist(k) = sum((model1.spheres.centers(i,:) - model2.spheres.centers(j,:))^2,2) ./ (model1.spheres.radius(i) + model2.spheres.radius(j))^2;
% end
% end