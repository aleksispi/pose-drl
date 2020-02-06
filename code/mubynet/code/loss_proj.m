function [f, g, stats] = loss_proj(T, angle, x3d, x2d, cx, cy, scale_ratio, calib)

np = size(T, 1);

focal = get_focal_from_angle(angle);

if nargin >= 6
    fx = focal * cx;
    fy = focal * cy;
end

% if nargin == 8
%     K = calib.K;
%     fx = K(1, 1) / scale_ratio;
%     fy = K(2, 2) / scale_ratio;
%     cx = K(1, 3);
%     cy = K(2, 3);
% end
    
N = size(x3d, 2);

% keyboard;
x3d = x3d + repmat(reshape(T, [np 1 3]), [1 N 1]);

proj = zeros(np, N, 2);
invZ = 1 ./ x3d(:, :, 3);
nX = x3d(:, :, 1) .* invZ;
nY = x3d(:, :, 2) .* invZ;
proj(:, :, 1)  = fx * nX + cx;
proj(:, :, 2)  = fy * nY + cy;

%     keyboard;
f = sum(sum(x2d(:, :, 3) .* sqrt(sum((proj - x2d(:, :, 1:2)).^2, 3)+1e-7)));
f = f / N / np;

if(nargout >= 2)
    dfdproj = (repmat(x2d(:, :, 3), [1 1 2])  .* (proj - x2d(:, :, 1:2)) ./ repmat(sqrt(sum((proj - x2d(:, :, 1:2)).^2, 3)+1e-7), [1 1 2])) / N / np;
    
    dfdx3d = zeros(size(x3d));
    dfdproj(:, :, 1) = fx * dfdproj(:, :, 1);
    dfdproj(:, :, 2) = fy * dfdproj(:, :, 2);
    
    dfdx3d(:, :, 1) = dfdproj(:, :, 1) .* invZ;
    dfdx3d(:, :, 2) = dfdproj(:, :, 2) .* invZ;
    dfdx3d(:, :, 3) = (-dfdx3d(:, :, 1) .* nX  - dfdx3d(:, :, 2) .* nY);
    
    dfdT = sum(dfdx3d, 2);
    dfdT = dfdT(:);
   
    g = dfdT;
end
if(nargout == 3)
    stats.proj = proj;
    stats.focal = [fx fy cx cy];
    stats.x3d = x3d;
end
end



function focal = get_focal_from_angle(angle)
focal = 1 ./ tan(angle * 0.5);
end
