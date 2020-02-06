classdef NoopStat < Stat
    properties
        mean
        means
        ma
        mas
    end
    
    methods
        function obj = NoopStat(collector, name, opts)
            default_opts = struct();
            obj = obj@Stat(collector, name, opts, default_opts);
            
            % Set dummy values.
            obj.mean = 0;
            obj.ma = 0;
            obj.mas = [0];
            obj.means = [0];
        end
        
        function collect_(obj, value)
        end
    end
    
end

