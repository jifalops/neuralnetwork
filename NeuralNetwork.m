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
        % Full set of input and output data used for training or validation
        trainingData;
        validationData;
        
        % Number of inputs, outputs, and hidden neurons
        nInputs;
        nOutputs;       
        nHidden;
        
        % Several Three-dimensional matrices of outputs and their respective errors
        % for both training and validation.
        % 1 - Each is a sample (only one for batch mode)
        % 2 - Each is an epoch
        % 3 - Each is an output neuron        
        trainingOutputs;        
        trainingErrors;
        
        validationOutputs;        
        validationErrors;        
        
        % Four dimensional matrix where each item in the fourth dimension is
        % an instance of a three dimensional weight matrix. These apply 
        % only to training, not validation. Weight matrices are described 
        % in further detail below.
        weightHistory;    
        derivativeHistory;    
        
        hasTrained;
    end
    
    methods
        
        %%
        % train()   - Train this neural network using the back-propagation algorithm.
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
        function [this weights] = train(this, data, nInputs, nHidden, initialWeights)
            
            % ===================================
            % Validate input parameters/arguments
            % and set initial state.
            % ===================================                        
            
            % data
            if nargin < 2 || isempty(data) 
                error('train(): Cannot train without data.');
            end
            
            % nInputs
            if nargin < 3 || nInputs < 1
               error('train(): There must be at least one input.');
            end            
            
                % nOutputs
                nOutputs = size(data, 2) - nInputs;
                if nOutputs < 1
                   error('train(): There must be at least one output.'); 
                end

                % nSamples
                nSamples = size(data, 1);
                if nSamples < 1
                    error('train(): There must be at least one sample.');
                end
                
                % inputs
                inputs = data(:, 1:nInputs);
                
                % outputs
                outputs = data(:, (nInputs + 1):(nInputs + nOutputs));
            
            % nHidden
            if nargin >= 4 && nHidden < 1
                error('train(): There must be at least one hidden neuron.');
            else
                % Default number of hidden neurons is the average number of 
                % inputs and outputs, rounded up.
                nHidden = ceil(mean([nInputs nOutputs]));
            end
            
            % initialWeights
            weights = this.makeWeightMatrix(nInputs, nHidden, nOutputs);            
            if nargin >= 5 
                if size(initialWeights) ~= size(weights)
                    error('train(): Size of weight matrix should be %s.', size(weights));
                else
                    weights = initialWeights;
                end               
            end
            
            
            % 
            % Reset information about previous training/validation attempts
            % 
            
            this.trainingData      = data;
            this.validationData    = null(1);
            
            this.nInputs           = nInputs;
            this.nOutputs          = nOutputs;
            this.nHidden           = nHidden;            
                        
            this.trainingOutputs   = null(1);
            this.trainingErrors    = null(1);
            
            this.validationOutputs = null(1);
            this.validationErrors  = null(1);
            
            this.weightHistory     = zeros(size(weights));
            this.derivativeHistory = zeros(size(weights));
                        
            hasTrained             = 0;
            
            % ===========================================================
            % Train using the specified mode and limits of this instance.
            % ===========================================================
            
            iEpoch = 1;
            outputError = Inf;   
            switch this.trainingMode
                case NeuralNetwork.TRAINING_MODE_SAMPLE_BY_SAMPLE
                    while ~(this.isComplete(iEpoch, outputError))
                        
                        % For each sample
                        for jSample = 1 : nSamples
                            sampleInputs = inputs(jSample, :);
                            sampleOutputs = outputs(jSample, :);
                            
                            % Calculate ynn
                            [this computedOutputs z] = this.computeOutputs(sampleInputs, weights, nOutputs);
                            
                            % Calculate the error of the outputs
                            [this outputError] = this.computeError(computedOutputs, sampleOutputs);                            
                                                                                    
                            % Compute the error derivatives for each weight
                            [this derivatives] = this.computeDerivatives(computedOutputs, ...
                                sampleInputs, sampleOutputs, weights, z);
                            
                            % Update the weight values
                            [this weights] = this.updateWeights(weights, derivatives, nInputs, nOutputs);
                            
                            
                            % Add to errorHistory
                            this.errorHistory(jSample, iEpoch) = outputError;
                            
                            % Add to outputHistory
                            for k = 1 : nOutputs
                                this.outputHistory(jSample, ((iEpoch - 1) * k) + k)
                            end
                        end
                        iEpoch = iEpoch + 1;
                    end
       
                    for iSample = 1 : nSamples
                       %subplot(nSamples, 1, iSample), 
                       plot(this.trainingErrors(iSample, :)); 
                    end
                case NeuralNetwork.TRAINING_MODE_BATCH
                    error('train(): Batch mode is not implemented yet.');
            end

            
            
            
            % calculate output using intputs and weights
        end
    end
    
    methods (Access = private)
        
        %%
        function [this weights] = makeWeightMatrix(this, nInputs, nHidden, nOutputs)
            x = nHidden;
            y = max(nInputs, nOutputs);
            z = 4;
            
            weights = zeros(x, y, z);            
            
            for jHidden = 1 : nHidden   
               % Input-Hidden link default weights (layer 1)
               for iInput = 1 : nInputs
                    weights(jHidden, iInput, 1) = (-1)^(iInput + jHidden);
               end
               
               % Hidden bias weights (layer 2)
               weights(jHidden, 1, 2) = 1;
            end
            
            for jHidden = 1 : nHidden
               for kOutput = 1 : nOutputs
                    
                   % Hidden-Output link default weights (layer 3)
                    weights(jHidden, kOutput, 3) = (-1)^(jHidden + kOutput);
                    
                    % Output bias weights (layer 4)
                    if jHidden == 1
                        weights(1, kOutput, 4) = 1;
                    end                    
               end               
            end
            
        end
         
        %%
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
        
        %% Compute output(s) of one sample
        function [this y z] = computeOutputs(this, inputs, weights, nOutputs)
            nInputs = size(inputs, 2);
            nHidden = size(weights, 1);
                        
            gamma = zeros(nHidden, 1);
            z     = zeros(nHidden, 1);  
            y     = zeros(nOutputs, 1);
            
            for kOutput = 1 : nOutputs
                
                % Initial output value is it's bias                
                y(kOutput) = weights(1, kOutput, 4);
                
                for jHidden = 1 : nHidden                    
                    
                    % Initial gamma value is the hidden neuron's bias
                    gamma(jHidden) = weights(jHidden, 1, 2);
                    
                    % Add each term together for gamma
                    for iInput = 1 : nInputs                        
                        gamma(jHidden) = gamma(jHidden) + weights(jHidden, iInput, 1) * inputs(iInput);
                    end
                    
                    % Calculate z for this hidden neuron
                    z(jHidden) = 1 / (1 + exp(-gamma(jHidden)));
                    
                    % Add this hidden neuron's effect on the current output
                    y(kOutput) = y(kOutput) + weights(jHidden, kOutput, 3) * z(jHidden);
                end          
            end
        end
        
        %% Compute the error of one sample
        function [this err] = computeError(this, y, outputs)
            err = 0;
            for iOutput = 1 : length(outputs)
               err = err + (y(iOutput) - outputs(iOutput))^2;
            end
            err = err / 2;
        end
        
        %%
        function [this derivatives] = computeDerivatives(this, y, inputs, ...
                outputs, weights, z)
            
            nInputs = size(inputs, 2);
            nHidden = size(weights, 1);
            nOutputs = size(outputs, 2);
            
            derivatives = zeros(size(weights));
            
            for kOutput = 1 : nOutputs
                
                % Derivative with respect to the output
                % ynn - ytable
                derivatives(1, kOutput, 4) = y(kOutput) - outputs(kOutput);
                
                for jHidden = 1 : nHidden;
                    
                    % Derivative with respect to hidden-output link weights
                    % dE/dy * zj
                    derivatives(jHidden, kOutput, 3) = derivatives(1, kOutput, 4) * z(jHidden);
                    
                    % Derivative with respect to hidden neuron biases
                    % sum(dE/dy * (hidden-output link derivative) * zj * (1 - zj))
                    derivatives(jHidden, 1, 2) = derivatives(jHidden, 1, 2) ...
                        + derivatives(1, kOutput, 4) ...
                        * derivatives(jHidden, kOutput, 3) ...
                        * z(jHidden) * (1 - z(jHidden));
                    
                    % Derivative with respect to input-hidden link weights
                    % sum(dE/dy * (hidden-output link derivative) * zj * (1 - zj) * xi)
                    for iInput = 1 : nInputs
                        derivatives(jHidden, iInput, 1) = derivatives(jHidden, iInput, 1) ...
                            + derivatives(1, kOutput, 4) ...
                            * derivatives(jHidden, kOutput, 3) ...
                            * z(jHidden) * (1 - z(jHidden)) * inputs(iInput);
                    end
                end
            end
        end
        
        %%
        function [this weights] = updateWeights(this, weights, derivatives, nInputs, nOutputs)            
            nHidden = size(weights, 1);            
                 
            % Update input-hidden link weights
            for iInput = 1 : nInputs                
                for jHidden = 1 : nHidden
                    weights(jHidden, iInput, 1) = weights(jHidden, iInput, 1) ...
                        - derivatives(jHidden, iInput, 1) * this.alpha;
                end
            end
            
            
            for jHidden = 1 : nHidden
                
                % Update hidden neuron biases
                weights(jHidden, 1, 2) = weights(jHidden, 1, 2) ...
                    - derivatives(jHidden, 1, 2) * this.alpha;

                % Update hidden-output link weights
                for kOutput = 1 : nOutputs                        
                    weights(jHidden, kOutput, 3) = weights(jHidden, kOutput, 3) ...
                        - derivatives(jHidden, kOutput, 3) * this.alpha;                    
                end
            end
            
             % Update output neuron biases
            for kOutput = 1 : nOutputs               
                weights(1, kOutput, 4) = weights(1, kOutput, 4) ...
                    - derivatives(1, kOutput, 4) * this.alpha;  
            end
            
        end
        
        %%
        function [this validation] = verror(inputs, outputs)
           for i =  1 : size(outputs, 1)
              for j = 1 : size(outputs, 2)
                  outputs(i, j) * inputs(i, ) - outputs(i,j)
              end
           end
        end
    end
end