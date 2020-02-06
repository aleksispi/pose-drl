function X = castG(X, varargin)
    X = cast(X, varargin{1});
    if(numel(varargin) == 2)
        X = gpuArray(X);
    end
end