classdef LearningRate < Stat
    % A special type of statistic that tracks the learning rate.
    
    properties
        lr
        lr_update_frac
        lr_update_steps
        lr_start
        lr_step_idx
    end
    
    methods
        
        function obj = LearningRate(collector, name, opts)
            global CONFIG
            default_opts = struct;
            obj = obj@Stat(collector, name, opts, default_opts);
            
            obj.lr_start = CONFIG.training_agent_lr;
            obj.lr = obj.lr_start;
            obj.lr_update_frac = CONFIG.training_agent_lr_update_factor;
            obj.lr_update_steps = CONFIG.training_agent_lr_update_steps;
            obj.lr_step_idx = 1;
        end
        
        function next_batch(obj)
            if obj.lr_step_idx <= numel(obj.lr_update_steps) && ...
               rem(obj.stats.batch_idx, obj.lr_update_steps(obj.lr_step_idx)) == 0
                obj.lr = obj.lr * obj.lr_update_frac;
                obj.lr_step_idx = obj.lr_step_idx + 1;
            end
        end
        
        function collect_(obj, value)
        end
        
        function print(obj)
            fprintf('%-40s cur: %10.10f\n', obj.name, obj.lr);
        end
    end
end