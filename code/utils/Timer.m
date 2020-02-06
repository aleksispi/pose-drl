
classdef Timer < handle
    %TIMER Pratical tool for timing stuff
    
    properties
        name
        timers
    end
    
    methods
        
        function obj = Timer(name)
            obj.name = name;
            obj.timers = struct;
        end
        
        function tic(obj, name)
            id = obj.slug(name);
            if ~isfield(obj.timers, id)
                obj.timers.(id).average = 0;
                obj.timers.(id).counter = 0;
                obj.timers.(id).name = name;
            end
            obj.timers.(id).current = tic;
        end
        
        function toc(obj, name)
            id = obj.slug(name);
            if ~isfield(obj.timers, id) || isnan(obj.timers.(id).current)
                error(strcat('Timers.toc called before tic for %s', name));
            end
            
            stop_time = toc(obj.timers.(id).current);
            obj.timers.(id).average = ...
                (obj.timers.(id).counter ...
                 * obj.timers.(id).average + stop_time) ...
                / (obj.timers.(id).counter + 1);
            obj.timers.(id).counter = obj.timers.(id).counter + 1;
            
            % Invalidate start time
            obj.timers.(id).current = nan;
        end
        
        function avg = avg_time(obj, name)
            id = obj.slug(name);
            if ~isfield(obj.timers, id)
                error(strcat('Unregistered timer %s', name));
            end
            avg = obj.timers.(id).average;
        end
        
        function print_avg_times(obj)
            fprintf('%s\n', obj.name);
            timer_ids = fieldnames(obj.timers);
            for idx = 1:length(timer_ids)
                timer = obj.timers.(timer_ids{idx});
                fprintf('%-40s avg: %.4f sec\n', timer.name, timer.average);
            end
            fprintf('\n');
        end
        
        function s = slug(~, s)
            s = regexprep(s, '[^A-Za-z_]' , '_');
        end
    end
end