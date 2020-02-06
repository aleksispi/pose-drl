classdef HistogramStat < Stat
    %HISTOGRAMSTAT Tracks the distribution of integer values using a
    % histogram.
    
    properties
        hist
        count
    end
    
    methods
        function obj = HistogramStat(collector, name, opts)
            default_opts = struct('max_bins_to_print', 10, ...
                                  'sort_bins_by', 'value'); % value / count
            obj = obj@Stat(collector, name, opts, default_opts);
            
            obj.hist = struct;
            obj.count = 0;
        end
        
        function collect_(obj, value)
            key = obj.stats.slug(sprintf('v%010d', value));
            if isfield(obj.hist, key)
                obj.hist.(key).count = obj.hist.(key).count + 1;
            else
                obj.hist.(key).value = value;
                obj.hist.(key).count = 0;
            end
            obj.count = obj.count + 1;
        end

        function print(obj)
            % Get all values
            bins = obj.get_sorted_bins();
            
            % Print limited nbr of bins
            nbr_print = min(obj.opt('max_bins_to_print'), length(bins));
            bins = bins(1:nbr_print);
            
            % Setup strings
            bin_string = sprintf('%-41s|', obj.name);
            value_string = sprintf('          %30s |', sprintf('(%d)', obj.count));
            
            % Print bins and dist
            for idx = 1:length(bins)
               bin = bins{idx};
               bin_string = strcat(bin_string, sprintf(' %-6d |', bin.value));
               value_string = strcat(value_string, sprintf(' %-6.2f |', bin.count / obj.count));
            end

            fprintf('%s\n%s\n', bin_string, value_string);
        end
        
        function plot(obj, save_path)
            if ~obj.opt('save_plot') && ~obj.opt('show_plot')
                % Neither save nor show plot, as such skip it
                return
            end
            
            % Get all values
            bins = obj.get_sorted_bins();

            x = zeros(size(bins));
            y = zeros(size(bins));
            
            % Print bins and dist
            for idx = 1:length(bins)
               bin = bins{idx};
               x(idx) = bin.value;
               y(idx) = bin.count;
            end            

            fig = figure('Visible', 'off');
            h = bar(x, y);
            title(obj.name, 'Interpreter', 'none');

            file_path = obj.file_save_path(save_path, '.png');
            if obj.opt('save_plot'); saveas(fig, file_path); end
            if obj.opt('show_plot'); figure(fig); end
        end
        
        function bins = get_sorted_bins(obj)
            % Retrieves the keys for the bins sorted according to the
            % opt: "sort_bins_by".
            if strcmp(obj.opt('sort_bins_by'), 'count')
                cell_struct = struct2cell(obj.hist);
                [~, I] = sort(cellfun (@(x) x.count, cell_struct), 'descend');
                bins = cell_struct(I);
            else
                % By value
                cell_struct = struct2cell(obj.hist);
                [~, I] = sort(cellfun (@(x) x.value, cell_struct));
                bins = cell_struct(I);
            end
        end
        
    end
    
end