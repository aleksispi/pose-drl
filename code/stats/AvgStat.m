classdef AvgStat < Stat
    % This is a basic stat that calculate ma and mean for a stat
    
    properties
        mean
        ma
        means
        mas
        ma_weight
        counter
    end
    
    methods
        
        function obj = AvgStat(collector, name, opts)
            default_opts = struct('force_save_hist', 0);
            obj = obj@Stat(collector, name, opts, default_opts);
            
            obj.counter = 0;
            obj.mean = 0;
            obj.ma = nan;
            obj.means = [];
            obj.mas = [];
        end
        
        function collect_(obj, value)
            obj.counter = obj.counter + 1;
            obj.mean = obj.calc_mean(obj.mean, value, obj.counter);
            
            if isnan(obj.ma)
                obj.ma = value;
            else
                if numel(obj.mas) < 500
                    obj.ma = obj.mean;
                else
                    obj.ma = obj.calc_ma(obj.ma, value, obj.opt('ma_weight'));
                end
            end
            
            if obj.opt('force_save_hist') || obj.opt('save_plot') || ...
                    obj.opt('show_plot')
                obj.means = [obj.means, obj.mean];
                obj.mas = [obj.mas, obj.ma];
            end
        end
        
        function data = get_data(obj)
            data = struct('means', obj.means, 'mas', obj.mas);
        end
        
        function plot(obj, save_path)
            if ~obj.opt('save_plot') && ~obj.opt('show_plot')
                % Neither save nor show plot, as such skip it
                return
            end

            series = {struct('x', 1:numel(obj.means), ...
                             'y', obj.means, ...
                             'legend_str', 'Mean', ...
                             'color_spec', '-r'), ...
                      struct('x', 1:numel(obj.mas), ...
                             'y', obj.mas, ...
                             'legend_str', 'Moving Average', ...
                             'color_spec', '-g')};
            
            plot_structs = {struct(...
                'title', obj.name, ...
                'xlabel', 'Step', ...
                'ylabel', '', ...
                'series', {series})};

            fig = obj.plot_values(plot_structs);

            file_path = obj.file_save_path(save_path, '.png');
            if obj.opt('save_plot')
                saveas(fig, file_path)
            end
            if obj.opt('show_plot')
                figure(fig)
            end
        end
        
        function print(obj)
            if numel(obj.mean) == 1
                fprintf('Mean %-35s tot: %10.5f, ma: %10.5f\n', obj.name, obj.mean, obj.ma);
            else
                fprintf('Mean %-35s tot: (%.2f, ', obj.name, obj.mean(1));
                for i = 2 : numel(obj.mean) - 1
                    fprintf('%.2f, ', obj.mean(i));
                end
                fprintf('%.2f) \n                                         ma:  (%.2f, ', obj.mean(end), obj.ma(1));
                for i = 2 : numel(obj.ma) - 1
                    fprintf('%.2f, ', obj.ma(i));
                end
                fprintf('%.2f)\n', obj.ma(end));
            end
        end 
    end
end
