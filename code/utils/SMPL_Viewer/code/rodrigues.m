function [R, dR] = rodrigues(r, mode)

% if(~exist('mode', 'var'))
%     mode = {'double'};
% end
% if(strcmp(mode{1}, 'double'))
%     [R, dR] = rodrigues_double(r);
%     return;
% end
assert(numel(r) == 3);
r = reshape(r, [3 1]);
theta = norm(r);
% dR = zeros(3, 9, mode{:}); % 3x9
dR = [0 0 0 0 0 0 0 0 0; ...
      0 0 0 0 0 0 0 0 0; ...
      0 0 0 0 0 0 0 0 0];
jacobian = (nargout == 2);

R = eye(3);

if(theta < eps)
    if(jacobian)
        dR([6 16 20]) = -1;
        dR([8 12 22]) = 1;
    end
    % put in column-major form
    dR = reshape(dR, [3 3 3]);
    dR = permute(dR, [1 3 2]);
    dR = reshape(dR, [3 9]);
    return;
end

if(theta)
    itheta = 1./theta;
else
    itheta = 0;
end
c = cos(theta);
s = sin(theta);
c1 = 1 - c;

r = r * itheta;
rrt = r*r';
r_x = [ 0 -r(3) r(2); r(3) 0 -r(1); -r(2) r(1) 0];

R = c*R + c1*rrt + s * r_x;
dR = [];


if(jacobian)
    
%     I = zeros(1, 9, mode{:});
%     drrt = zeros(3, 9, mode{:});
%     d_r_x_ = zeros(3, 9, mode{:});
    
%     I(:) = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    
   
%     keyboard;
    d_r_x_ = [0, 0, 0, 0, 0, -1, 0, 1, 0;...
        0, 0, 1, 0, 0, 0, -1, 0, 0;...
        0, -1, 0, 1, 0, 0, 0, 0, 0];
    
   
%    
    
    rrt = reshape(rrt', 1, 9);
    r_x = reshape(r_x', 1, 9);
    
    
    a2 = c1 * itheta;
    a4 = s * itheta;
    a0 = -s * r;
    a1 = (s - a2 - a2)*r;
    a3 = (c - a4)*r;
    
%     keyboard;
    r = r * a2;
    drrt = [r(1)+r(1), r(2), r(3), r(2), 0, 0, r(3), 0, 0; ...
        0, r(1), 0, r(1), r(2)+r(2), r(3), 0, r(3), 0; ...
        0, 0, r(1), 0, 0, r(2), r(1), r(2), r(3)+r(3)];
     
%     keyboard;
    dR = a1 * rrt + drrt + a3*r_x + a4*d_r_x_;
    
    dR(:, 1) = dR(:, 1) + a0;
    dR(:, 5) = dR(:, 5) + a0;
    dR(:, 9) = dR(:, 9) + a0;
    
    % put in column-major form
    dR = reshape(dR, [3 3 3]);
    dR = permute(dR, [1 3 2]);
    dR = reshape(dR, [3 9]);
end

end

function [R, dR] = rodrigues_double(r)

assert(numel(r) == 3);
r = reshape(r, [3 1]);
theta = norm(r);
dR = zeros(3, 9); % 3x9
jacobian = (nargout == 2);

% o_ = ones(1, 1, mode{:});
% z_ = zeros(1, 1, mode{:});

if(theta < eps)
    R = eye(3);
    if(jacobian)
        dR([6 16 20]) = -1;
        dR([8 12 22]) = 1;
    end
    % put in column-major form
    dR = reshape(dR, [3 3 3]);
    dR = permute(dR, [1 3 2]);
    dR = reshape(dR, [3 9]);
    return;
end

if(theta)
    itheta = 1./theta;
else
    itheta = 0;
end
c = cos(theta);
s = sin(theta);
c1 = 1 - c;

r = r * itheta;
rrt = r*r';
r_x = [ 0 -r(3) r(2); r(3) 0 -r(1); -r(2) r(1) 0];

R = c*eye(3) + c1*rrt + s * r_x;
dR = [];


if(jacobian)
    
    I = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    drrt = [r(1)+r(1), r(2), r(3), r(2), 0, 0, r(3), 0, 0; ...
        0, r(1), 0, r(1), r(2)+r(2), r(3), 0, r(3), 0; ...
        0, 0, r(1), 0, 0, r(2), r(1), r(2), r(3)+r(3)];
    d_r_x_ = [0, 0, 0, 0, 0, -1, 0, 1, 0;...
        0, 0, 1, 0, 0, 0, -1, 0, 0;...
        0, -1, 0, 1, 0, 0, 0, 0, 0];
%     keyboard;
    rrt = reshape(rrt', 1, 9);
    r_x = reshape(r_x', 1, 9);
    
    a0 = -s * r;
    a1 = (s - 2*c1*itheta)*r;
    a2 = repmat(c1 * itheta, 3, 9);
    a3 = (c - s*itheta)*r;
    a4 = repmat(s*itheta, 3, 9);
    
    
    dR = a0 * I + a1 * rrt + a2.*drrt + a3*r_x + a4.*d_r_x_;
    % put in column-major form
    dR = reshape(dR, [3 3 3]);
    dR = permute(dR, [1 3 2]);
    dR = reshape(dR, [3 9]);
end

end