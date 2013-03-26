%% Utility functions
classdef Util < handle
   %%
   methods (Static)
       %% 
       % Scale and randomize data (matrix or filename). Use for
       % convenience instead of individual functions.
       function [data outfile] = scaleAndRandomize(data, numOutputs, inputMinValue, inputMaxValue, outputMinValue, outputMaxValue)
           if nargin < 2
               error('scaleAndRandomize(): Not enough input arguments.');
           end
           
           outfile = '';           
           if ischar(data)
              outfile = Util.appendFilename(data, '_scaled_randomized');
              data = xlsread(data);
           end
           
           switch nargin
               case 2
                   data = Util.scale(data, numOutputs);
               case 4
                   data = Util.scale(data, numOutputs, inputMinValue, inputMaxValue);
               case 6
                   data = Util.scale(data, numOutputs, inputMinValue, inputMaxValue, outputMinValue, outputMaxValue);
               otherwise
                   error('scaleAndRandomize(): Invalid number of arguments.');
           end
           
           data = Util.randomize(data);
           
           if ~isempty(outfile)
              xlswrite(outfile, data);
           end
       end
       
       %% 
       % Scale `data` (a matrix or an xls filename) to the given min/max values.
       % default input limits are [-1, 1] default output limits are [0, 1].
       % If data is a filename, filename_scaled.xls will also be written.
       function [scaled outfile] = scale(data, numOutputs, inputMinValue, inputMaxValue, outputMinValue, outputMaxValue) 
           if nargin < 2
               error('scale(): Not enough input arguments.');
           end
       
           outfile = '';           
           if ischar(data)
              outfile = Util.appendFilename(data, '_scaled');
              data = xlsread(data);
           end
           
           if isempty(data)
              error('scale(): Data is empty.'); 
           end
           
           if numOutputs < 1
              error('scale(): There must be at least one output.'); 
           end
           
           numInputs = size(data, 2) - numOutputs;
           
           if numInputs < 1
              error('scale(): There must be at least one input.'); 
           end
           
           numSamples = size(data, 1);
           
           if nargin == 3 || nargin == 5
              error('scale(): maxValue must be defined if minValue is defined.'); 
           end
           
           if nargin == 4 || nargin == 6
               if inputMinValue >= inputMaxValue || outputMinValue >= outputMaxValue
                    error('scale(): minValue must be less than maxValue.');                
               end    
           else
               inputMinValue  = -1;
               inputMaxValue  = 1;
               outputMinValue = 0;
               outputMaxValue = 1;
           end
              
           scaled = zeros(size(data));
           
           scaleDiff = inputMaxValue - inputMinValue;           
           for i = 1 : numInputs
              minval = min(data(:, i));
              maxval = max(data(:, i)); 
              dataDiff = maxval - minval;
              for j = 1 : numSamples
                 scaled(j, i) = inputMinValue + ((data(j, i) - minval) * scaleDiff) / dataDiff;   
              end
           end
           
           scaleDiff = outputMaxValue - outputMinValue;
           for i = (numInputs + 1) : (numInputs + numOutputs)
              minval = min(data(:, i));
              maxval = max(data(:, i)); 
              dataDiff = maxval - minval;
              for j = 1 : numSamples
                 scaled(j, i) = outputMinValue + ((data(j, i) - minval) * scaleDiff) / dataDiff;   
              end
           end
           
           if ~isempty(outfile)
              xlswrite(outfile, scaled);
           end
       end
       
        %%
        % Randomize the rows in `data` (a matrix or an xls filename).
        % If data is a filename, filename_randomized.xls will also be written.
        function [randomized outfile] = randomize(data)
            if nargin < 1
                error('randomize(): Not enough input arguments.');
            end

            outfile = '';
            if ischar(data)
                outfile = Util.appendFilename(data, '_randomized');
                data = xlsread(data);
            end

            if isempty(data)
                error('randomize(): Data is empty.'); 
            end

            randomized = data(randperm(size(data,1)),:);
            
            if ~isempty(outfile)
              xlswrite(outfile, randomized);
           end
        end
       
        %%
        % Insert text between the end of a file's name and the extension.
        function newname = appendFilename(filename, text)
            if nargin < 2 || ~ischar(filename) || ~ischar(text)
                error('appendFilename(): Two string arguments are required.'); 
            end
            strcat(filename, '');
            [~, ~, ext] = fileparts(filename);
            newname = strcat(filename(1:(length(filename) - length(ext))), text, ext);           
        end
   end
   
   methods (Access = private)
       
   end
end
