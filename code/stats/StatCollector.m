classdef StatCollector < handle
    %STATCOLLECTOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        stats
        name
        epoch_idx
        ep_idx
        batch_idx
        step_idx
    end
    
    methods
        
        function obj = StatCollector(name)
            addpath('code/stats');
            obj.stats = struct;
            obj.name = name;
            
            obj.batch_idx = 1;
            obj.ep_idx = 1;
            obj.step_idx = 1;
            obj.epoch_idx = 1;
        end
        
        function tf = has_stat(obj, name)
            tf = isfield(obj.stats, obj.slug(name));
        end
        
        function register(obj, type, name, opts)
            
            global CONFIG
            % Check if stat is disabled in config, then replace with noop
            if any(strcmp(CONFIG.stats_disabled_stats, name))
                type = 'noop';                
            end
            
            if obj.has_stat(name)
                error('Stat already exists');
            end
            
            if ~exist('opts', 'var')
                opts = struct;
            end
            
            if strcmp(type, 'avg')
                cls = AvgStat(obj, name, opts);
            elseif strcmp(type, 'current')
                cls = CurrentStat(obj, name, opts);
            elseif strcmp(type, 'lr')
                cls = LearningRate(obj, name, opts);
            elseif strcmp(type, 'hist')
                cls = HistogramStat(obj, name, opts);
            elseif strcmp(type, 'noop')
                cls = NoopStat(obj, name, opts);
            else    
                error('Unknown stat type');
            end
            
            stat = struct('class', cls, 'name', name, 'type', type, 'opts', opts);
            obj.stats.(obj.slug(name)) = stat;
        end
        
        function cls = s(obj, name)
            key = obj.slug(name);
            if ~isfield(obj.stats, key)
                fprintf('User accessed invalid stat: %s, adding NoopStat\n', name);
                obj.register('noop', name);
            end
            cls = obj.stats.(key).class;
        end
        
        function collect(obj, name_value_array)
            for idx = 1:2:length(name_value_array)
                stat = obj.s(name_value_array{idx});
                value = name_value_array{idx + 1};
                stat.collect(value);
            end
        end
        
        function next_step(obj)
            obj.step_idx = obj.step_idx + 1;
            stat_names = fieldnames(obj.stats);
            for idx = 1:length(stat_names)
               stat = obj.stats.(stat_names{idx});
               stat.class.next_step()
            end
        end
        
        function next_batch(obj)
            obj.batch_idx = obj.batch_idx + 1;
            stat_names = fieldnames(obj.stats);
            for idx = 1:length(stat_names)
               stat = obj.stats.(stat_names{idx});
               stat.class.next_batch();
            end
        end
        
        function next_ep(obj)
            obj.ep_idx = obj.ep_idx + 1;
            stat_names = fieldnames(obj.stats);
            for idx = 1:length(stat_names)
               stat = obj.stats.(stat_names{idx});
               stat.class.next_ep()
            end
        end
        
        function next_epoch(obj)
            obj.epoch_idx = obj.epoch_idx + 1;
        end
        
        function print(obj)
            global CONFIG
            in_epoch = rem(obj.ep_idx, CONFIG.panoptic_nbr_time_freezes);
            fprintf('\n%s\nEpochs: %d (%d / %d), Episodes: %d / %d, Steps: %d, Batches: %d\n', ...
                    obj.name, obj.epoch_idx, in_epoch, CONFIG.panoptic_nbr_time_freezes, ...
                    obj.ep_idx, CONFIG.training_agent_nbr_eps, obj.step_idx, obj.batch_idx);

            stat_names = fieldnames(obj.stats);
            for idx = 1 : length(stat_names)
               stat = obj.stats.(stat_names{idx});
               stat.class.print()
            end

            fprintf('\n');
        end
            
        function plot(obj, save_path)
            stat_names = fieldnames(obj.stats);
            for idx = 1:length(stat_names)
               stat = obj.stats.(stat_names{idx});
               stat.class.plot(save_path)
            end
        end
        
        function s = slug(~, s)
            s = regexprep(s, '[^A-Za-z_\d]' , '_');
        end
        
        function nstat_clone = clone(obj, new_name)
            % Creates a new StatCollector with the same stats without any
            % data and with all counters resetted. 
            % Useful for cloning a set of registered stats.

            nstat_clone = StatCollector(new_name);
            stat_names = fieldnames(obj.stats);
            for idx = 1:length(stat_names)
               stat = obj.stats.(stat_names{idx});
               nstat_clone.register(stat.type, stat.name, stat.opts);
            end
        end

    end
    
end

