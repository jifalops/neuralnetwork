classdef NeuralNetwork < handle
    properties (Constant)             
       TRAINING_MODE_SAMPLE_BY_SAMPLE = 1;
       TRAINING_MODE_BATCH            = 2;       

       TERMINATION_MODE_NONE      = 0;
       TERMINATION_MODE_EPOCHS    = 1;
       TERMINATION_MODE_ERROR     = 2;
       TERMINATION_MODE_EITHER    = 3;
       TERMINATION_MODE_BOTH      = 4;
    end

    properties
        trainingMode        = NeuralNetwork.TRAINING_MODE_SAMPLE_BY_SAMPLE;
        terminationMode     = NeuralNetwork.TERMINATION_MODE_EITHER;          
        maxEpochs           = 100;
        maxError            = 0.01;
        alpha               = 0.1;
        %trackTrainingErrors = 1;
    end
    
    properties (SetAccess = private)
        % Three dimensional matrix of outputs, corresponds to errorHistory.        
        % x = Each "row"    is a sample (only one for batch mode)
        % y = Each "column" is an epoch
        % z = Each "layer"  is an output neuron
        outputHistory;
        
        % Three dimensional matrix of errors, corresponds to outputHistory.       
        % x = Each "row"    is a sample (only one for batch mode)
        % y = Each "column" is an epoch
        % z = Each "layer"  is an output neuron
        errorHistory;
        
        % Four dimensional matrix where each item in the fourth dimension is
        % an instance of a three dimensional weight matrix.
        weightHistory;
    end
    
    methods
        
        %%
        % train()   - Train the neural network using the given data. 
        %
        % Required parameters:
        %   data  - m by n matrix; m = samples, n = inputs and outputs
        %           (input columns and then output columns).
        %   nInputs - the number of inputs in the data table.      
        % Optional parameters:
        %   nHidden - Number of hidden neurons. Default is the mean number
        %             of inputs and output neurons rounded up (ceiling). 
        %   weights - The initial weights to use, given as a three dimensional
        %             matrix, x-y-z, where z always has the following layers:        
        %               1. Links between input and hidden neurons
        %               2. Hidden neuron biases
        %               3. Links between hidden and output neurons
        %               4. Output neuron biases        
        %             Each of these layers has its own sizes required to
        %             represent links using the x and y axes. For example,
        %             consider a neural network with 3 inputs, 10 hidden 
        %             neurons, and 2 outputs.
        %               Layer 1: x = 3, y = 10      (input, hidden)
        %               Layer 2: x = 10, y = 1      (hidden, none)
        %               Layer 3: x = 10, y = 2      (hidden, output)
        %               Layer 4: x = 2, y = 1       (output, none)
        %             The resultant matrix would be the largest of each
        %             dimension (10x10x4), while only 62 cells will be
        %             used. Memory usage could be optimized by ordering the 
        %             x and y values in each layer, but may not be worth the
        %             computational expense due to the frequent lookups
        %             needed.
        %             A better solution would be to designate positions for
        %             input, hidden, and output neurons. The method used
        %             here is to have the number of hidden neurons always
        %             be on the x axis; input and output neurons always
        %             reside on the y axis. For one dimensional layers, the
        %             number 1 is used for the type of neuron not being
        %             used. The resultant matrix is:
        %               Layer 1: x = 10, y = 3      (hidden, input)
        %               Layer 2: x = 10, y = 1      (hidden, none)
        %               Layer 3: x = 10, y = 2      (hidden, output)
        %               Layer 4: x = 1, y = 2       (none, output)
        %             10x3x4 or 120 is much closer to 62 than 400 was.
        %             There is no added computational expense because no 
        %             lookup is needed.
        function [this weights] = train(this, data, nInputs, nHidden, weights)
            
            % ===================================
            % Validate input parameters/arguments
            % ===================================
            
            % data
            if nargin < 2 || isempty(data) 
                error('train(): Cannot train without data.');
            end
            
            % nInputs
            if nargin < 3 || nInputs < 1
               error('train(): There must be at least one input.');
            end
            
            % nHidden
            if nargin >= 4 && nHidden < 1
                error('train(): There must be at least one hidden neuron.');
            end
            
            % weights
            if nargin >= 5 && isempty(weights)
                error('train(): There must be at least one hidden neuron.');
            end
            
            % Number of inputs, outputs, and samples.
            nInputs  = size(inputs, 2);
            nOutputs = size(outputs, 2);
            nSamples = size(inputs, 1);
                        
            if size(outputs, 1) ~= nSamples
                error('train(): Inputs and outputs must have the same number of samples.');
            end
            
            
            % Default number of hidden neurons is the average number of 
            % inputs and outputs, rounded up.
            if nHidden < 1
                nHidden = ceil(mean([nInputs nOutputs]));
            end

            % Number of weights required for this network.
            nWeights = nHidden * (nInputs + nOutputs + 1) + 1; 
            
            % Validate the number of weights given. If there are no weights
            % given, defaults will be provided.
            if ~isempty(weights) && length(weights) ~= nWeights                
                error('train(): Received %d weights, expected %d.', ...
                    length(weights), nWeights);  
            end
            
            % Get the weights in the format used internally (weight groups).
            % If initial weights were not given, default weights will be
            % returned.
            [this w1 w2 w3 w4] = this.getWeights(nInputs, nHidden, nOutputs, weights);
            
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            iEpoch = 1;
            currentError = Inf;            
            switch this.trainingMode
                case NeuralNetwork.TRAINING_MODE_SAMPLE_BY_SAMPLE
                    while ~(this.isComplete(iEpoch, currentError))
                        
                        % For each sample
                        for jSample = 1 : nSamples
                            
                            % Calculate ynn
                            [this y z] = this.computeOutput(inputs(jSample, :), w1, w2, w3, w4);
                            
                            % Calculate the error of ynn
                            [this currentError] = this.computeError(y, outputs(jSample, :));                            
                                                                                    
                            % Compute the error derivatives for each weight
                            [this e1 e2 e3 e4] = this.computeDerivatives(y, ...
                                inputs(jSample, :), outputs(jSample, :), w1, w2, w3, w4, z);
                            
                            % Update the weight values
                            [this w1 w2 w3 w4] = this.updateWeights(w1, w2, w3, w4, ...
                                e1, e2, e3, e4);
                            
                            
                            % Store the error for plotting later
                            this.trainingErrors(jSample, iEpoch) = currentError;
                            
                            for k = 1 : nOutputs
                                this.outputsComputed(jSample, ((iEpoch - 1) * k) + k)
                            end
                        end
                        iEpoch = iEpoch + 1;
                    end
       
                    for iSample = 1 : nSamples
                       %subplot(nSamples, 1, iSample), 
                       plot(this.trainingErrors(iSample, :)); 
                    end
                case NeuralNetwork.TRAINING_MODE_BATCH
            end

            
            
            
            % calculate output using intputs and weights
        end
    end
    
    methods (Access = private)
        
        function [this weights] = makeWeightMatrix(this, nInputs, nHidden, nOutputs)
            x = nHidden;
            y = max(nInputs, nOutputs);
            z = 4;
            weights = zeros(x, y, z);
        end
        
        % w1 = Weights for link between input & hidden layer.
        % w2 = Weights for bias of hidden layer.
        % w3 = Weights for link between hidden & output layer.
        % w4 = Weights for bias of hidden output.    
        function [this w1 w2 w3 w4] = getWeights(this, nInputs, nHidden, nOutputs, weights)
            
            % Create matrices of the correct size for each weight group            
            w1 = ones(nInputs, nHidden);
            w2 = ones(nHidden, 1);
            w3 = ones(nHidden, nOutputs);
            w4 = ones(nOutputs, 1);
            
            
            if isempty(weights)
                
                

                % w1 defaults
                for iInput = 1 : nInputs
                   for jHidden = 1 : nHidden
                        w1(iInput, jHidden) = (-1)^(iInput + jHidden);
                   end
                end

                % w3 defaults
                for jHidden = 1 : nHidden
                   for kOutput = 1 : nOutputs
                        w3(jHidden, kOutput) = (-1)^(jHidden + kOutput);
                   end
                end
                
            else                
                % A default set of weights was given in the form of an
                % array. Convert to the format used internally.
                
                % Hidden neuron biases
                start = nInputs * nHidden + 1;
                w2 = weights(start:(start + nHidden));
                
                % Output biases
                start = nWeights - nOutputs;
                w4 = weights(start:nWeights);
                
                % Weights between inputs and hidden layer
                for iInput = 1 : nInputs
                    for jHidden = 1 : nHidden          
                        index = (iInput - 1) * nHidden + jHidden;
                        w1(iInput, jHidden) = weights(index);
                    end
                end
                
                % Weights between hidden and output layer
                for jHidden = 1 : nHidden
                    for kOutput = 1 : nOutputs          
                        index = (jHidden - 1) * nOutputs + kOutput;
                        w3(jHidden, kOutput) = weights(index);
                    end
                end
            end
        end
        
        function [result this] = isComplete(this, iEpoch, currentError)
            switch this.terminationMode
                case NeuralNetwork.TERMINATION_MODE_NONE
                    result = 0;
                case NeuralNetwork.TERMINATION_MODE_EPOCHS
                    result = iEpoch > this.maxEpochs;
                case NeuralNetwork.TERMINATION_MODE_ERROR
                    result = currentError < this.maxError;
                case NeuralNetwork.TERMINATION_MODE_EITHER
                    result = iEpoch > this.maxEpochs ...
                             || currentError < this.maxError;
                case NeuralNetwork.TERMINATION_MODE_BOTH
                    result = iEpoch > this.maxEpochs ...
                             && currentError < this.maxError;
            end
        end
        
        % Compute output of one sample
        function [this y z] = computeOutput(this, inputs, w1, w2, w3, w4)
            nInputs = length(inputs);
            nHidden = length(w2);
            nOutputs = length(w4);
                        
            gamma = zeros(nHidden, 1);
            z     = zeros(nHidden, 1);  
            y     = zeros(nOutputs, 1);
            
            for iOutput = 1 : nOutputs
                y(iOutput) = w4(iOutput);
                for jHidden = 1 : nHidden
                    gamma(jHidden) = w2(jHidden);
                    for kInput = 1 : nInputs
                        gamma(jHidden) = gamma(jHidden) + w1(kInput, jHidden) * inputs(kInput);
                    end
                    z(jHidden) = 1 / (1 + exp(-gamma(jHidden)));
                    y(iOutput) = y(iOutput) + w3(jHidden, iOutput) * z(jHidden);
                end          
            end
        end
        
        % Compute the error of one sample
        function [this err] = computeError(this, y, outputs)
            err = 0;
            for iOutput = 1 : length(outputs)
               err = err + (y(iOutput) - outputs(iOutput))^2;
            end
            err = err / 2;
        end
        
        function [this e1 e2 e3 e4] = computeDerivatives(this, y, inputs, ...
                outputs, w1, w2, w3, w4, z)
            nInputs = length(inputs);
            nHidden = length(w2);
            nOutputs = length(w4);
            
            e1 = zeros(size(w1));
            e2 = zeros(size(w2));
            e3 = zeros(size(w3));
            e4 = zeros(size(w4));
            
            for iOutput = 1 : nOutputs
                e4(iOutput) = y(iOutput) - outputs(iOutput);
                for jHidden = 1 : nHidden;
                   e3(jHidden, iOutput) = e4(iOutput) * z(jHidden);
                   e2(jHidden) = e4(iOutput) * e3(jHidden, iOutput) ...
                       * z(jHidden) * (1 - z(jHidden));
                   for kInput = 1 : nInputs
                      e1(kInput, jHidden) = e4(iOutput) * e3(jHidden, iOutput) ...
                          * z(jHidden) * (1 - z(jHidden)) * inputs(kInput);
                   end
                end
            end
        end
        
        function [this w1 w2 w3 w4] = updateWeights(this, w1, w2, w3, w4, e1, e2, e3, e4)
            nInputs = size(w1, 1);
            nHidden = length(w2);
            nOutputs = length(w4);
                        
            for iInput = 1 : nInputs                
                for jHidden = 1 : nHidden
                    w1(iInput, jHidden) = w1(iInput, jHidden) - e1(iInput, jHidden) * this.alpha;
                    w2(jHidden) = w2(jHidden) - e2(jHidden) * this.alpha;                    
                    if iInput == 1
                        for kOutput = 1 : nOutputs                        
                            w3(jHidden, kOutput) = w3(jHidden, kOutput) - e3(jHidden, kOutput) * this.alpha;
                            w4(kOutput) = w4(kOutput) - e4(kOutput) * this.alpha; 
                        end
                    end
                end
            end
            
        end
        
        function [this validation] = verror(inputs, outputs)
           for i =  1 : size(outputs, 1)
              for j = 1 : size(outputs, 2)
                  outputs(i, j) * inputs(i, ) - outputs(i,j)
              end
           end
        end
    end
end