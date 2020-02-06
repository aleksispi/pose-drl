classdef CurrentStat < Stat
    % A special type of statistic that simply tracks the latest value
    
    properties
        value
    end
    
    methods
        function obj = CurrentStat(collector, name, opts)
            obj = obj@Stat(collector, name, opts, struct);
            obj.value = nan;
        end
        
        function collect_(obj, value)
            obj.value = value;
        end
        
        function print(obj)
            fprintf('%-40s cur: ', obj.name);
            for i = 1 : numel(obj.value)
                fprintf('%.4f, ', obj.value(i));
            end
            fprintf('\n');
        end
    end
end