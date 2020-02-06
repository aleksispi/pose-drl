function [f, dfdtheta] = GMMThetaPrior(theta, gmm, use_sum)

% keyboard;
gradient = nargout > 1;
x = theta(4:end);
x = reshape(x, 1, []);
gmm.covs = permute(gmm.covs, [2 3 1]);
ngauss = size(gmm.covs, 3);

% compute log-likelihoods
lls = sqrt(0.5) * x * reshape(gmm.covs, size(gmm.covs, 1), []);
lls = reshape(lls, [], ngauss);
lls = lls - gmm.opt_means;

% for i = 1 : ngauss
%     lls(i, :) = sqrt(0.5) * (x - gmm.means(i, :)) * gmm.covs(:, :, i);
% end

if(nargin < 3)
    % compute the minimum blob and forward
    llsc = sum(lls .^ 2, 1) - log(gmm.weights);
    [f, imin] = min(llsc);
else
    llsc = 0;
    gmm.weights = 1 ./ gmm.weights;
    nweights = sum(gmm.weights);
    gmm.weights = gmm.weights / nweights;
    for k = 1 : ngauss
        llsc = llsc + exp(sum(lls(k, :).^2, 2)) * gmm.weights(k);
    end
    llsc_log = log(llsc) + log(nweights);
    f = sum(llsc_log,2) / ngauss;
    if(isnan(f) || isinf(f))
        keyboard;
    end
end

% compute backward
if(gradient)
    if(nargin < 3)
        dfdtheta = 2 * sqrt(0.5) * gmm.covs(:, :, imin) * lls(:, imin);
        dfdtheta = [0 0 0 dfdtheta(:)'];
        dfdtheta = reshape(dfdtheta, size(theta));
    else
        dfdllsc = - (1/ngauss)./(llsc);
        dfdlls = zeros(size(lls));
        dfdtheta = zeros(size(x));
        for k = 1 : ngauss
            dfdlls(k, :) = -2 * gmm.weights(k) * lls(k, :) .* exp(sum(lls(k, :).^2, 2)) .* dfdllsc;
            dfdtheta = dfdtheta + sqrt(0.5) * dfdlls(k, :) * gmm.covs(:, :, k)';
        end
        dfdtheta = [0 0 0 dfdtheta];
        dfdtheta = reshape(dfdtheta, size(theta));
    end
end

end


