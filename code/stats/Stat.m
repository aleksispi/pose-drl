classdef Stat < handle
    % This is base version used for inheritance
    
    properties
        name
        stats
        opts
    end
    
    methods
        
        function obj = Stat(collector, name, overriding_opts, default_opts) 
            obj.name = name;
            obj.stats = collector;
            
            % Setups the opt hierarchy: 
            % overriding_opts > default_opts > global_opts
            global CONFIG
            obj.opts = struct('ma_weight', CONFIG.stats_ma_weight, ...
                              'save_plot', CONFIG.stats_save_plots, ...
                              'show_plot', CONFIG.stats_defaults_show_plots);
            
            for fn = fieldnames(default_opts)'
               obj.opts.(fn{1}) = default_opts.(fn{1});
            end
            for fn = fieldnames(overriding_opts)'
               obj.opts.(fn{1}) = overriding_opts.(fn{1});
            end
        end
        
        function collect(obj, value)
            obj.collect_(value);            
        end
        
        function collect_(obj, value)
            error('Not implemented');
        end
        
        function next_ep(obj)
        end
        
        function next_step(obj)
        end
        
        function next_batch(obj)
        end
        
        function plot(obj, save_path)
            % Mandatory function!
        end
        
        function print(obj)
            % Mandatory function!
        end
        
        function data = get_data(obj)
            % Returns a struct with data
            data = struct;
        end
        
        function new_mean = calc_mean(obj, old_val, value, idx)
            new_mean = ((idx - 1) * old_val + value) / idx;
        end
        
        function new_ma = calc_ma(obj, old_val, value, ma_weight)
            new_ma = (1 - ma_weight) * old_val + ma_weight * value;
        end
        
        function fig = plot_values(obj, plot_structs)
            % Takes a cell array of data structs and plots them into one
            % figure
            % fields:
            %   - xlabel
            %   - ylabel
            %   - title
            %   - series (cell array of structs:)
            %       - x
            %       - y
            %       - color_spec
            %       - legend_str
            
            nbr_plots =  numel(plot_structs);
            grid_size = ceil(sqrt(nbr_plots));
            
            fig = figure('Visible', 'off');
            
            hold on;
            for i = 1 : nbr_plots
                data = plot_structs{i};

                title(data.title, 'Interpreter', 'none');
                xlabel(data.xlabel, 'Interpreter', 'none');
                ylabel(data.ylabel, 'Interpreter', 'none');

                nbr_series = numel(data.series);
                
                subplot(grid_size, grid_size, i);
                legend_str = cell(nbr_series, 1);
                for j = 1 : nbr_series
                    serie = data.series{j};
                    p = plot(serie.x, serie.y, serie.color_spec);
                    legend_str{j} = serie.legend_str;
                end
                legend(legend_str);
            end
            hold off;            
        end
        
        function val = opt(obj, opt_name)
            if isfield(obj.opts, opt_name)
                val = obj.opts.(opt_name);
            else
                error('Unknown option');
            end
        end
        
        function file_path = file_save_path(obj, save_path, file_suffix)
            % Creates the correct path to save any plots or files from
            % stat. file_suffix should atleast contain the file ending and
            % may contain some unique identifier if a stat saves multiple
            % plots for example.
            file_path = sprintf('%s/%s_%s%s', save_path, obj.stats.slug(obj.stats.name), obj.stats.slug(obj.name), file_suffix);
        end
        
    end
    
end

