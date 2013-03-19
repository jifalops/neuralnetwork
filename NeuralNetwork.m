classdef NeuralNetwork < handle
    properties (Constant)    
       MODEL_TYPE_MLP3 = 1;     % Multi-Layer Perceptron
       MODEL_TYPE_RBF  = 2;     % Radial Basis Function
        
       TRAINING_MODE_SAMPLE_BY_SAMPLE = 1;
       TRAINING_MODE_BATCH            = 2;       

       TERMINATION_MODE_NONE      = 0;
       TERMINATION_MODE_EPOCHS    = 1;
       TERMINATION_MODE_ERROR     = 2;
       TERMINATION_MODE_EITHER    = 3;
       TERMINATION_MODE_BOTH      = 4;
       
       % Types of histories to remember. These can help analyze the
       % network, but may cost huge amounts of memory for large problems.
       HISTORY_TYPE_NONE               = bin2dec('000000');
       HISTORY_TYPE_TRAINING_OUTPUTS   = bin2dec('000001');
       HISTORY_TYPE_TRAINING_ERRORS    = bin2dec('000010');
       HISTORY_TYPE_VALIDATION_OUTPUTS = bin2dec('000100');
       HISTORY_TYPE_VALIDATION_ERRORS  = bin2dec('001000');
       HISTORY_TYPE_WEIGHTS            = bin2dec('010000');
       HISTORY_TYPE_DERIVATIVES        = bin2dec('100000');       
       % Combination histories
       HISTORY_TYPE_TRAINING           = bin2dec('000011');
       HISTORY_TYPE_VALIDATION         = bin2dec('001100');
       HISTORY_TYPE_OUTPUTS            = bin2dec('000101');
       HISTORY_TYPE_ERRORS             = bin2dec('001010');
       HISTORY_TYPE_ALL                = bin2dec('111111');
       
       WEIGHT_TYPE_I_PLUS_J = 1;
       WEIGHT_TYPE_RANDOM   = 2;
    end

    properties
        modelType           = NeuralNetwork.MODEL_TYPE_MLP3;
        weightType          = NeuralNetwork.WEIGHT_TYPE_I_PLUS_J;
        trainingMode        = NeuralNetwork.TRAINING_MODE_SAMPLE_BY_SAMPLE;
        terminationMode     = NeuralNetwork.TERMINATION_MODE_EITHER;          
        maxEpochs           = 100;
        maxError            = 0.01;
        alpha               = 0.1;
        histories           = NeuralNetwork.HISTORY_TYPE_ERRORS;
    end
    
    properties (SetAccess = private)                        
        % All of the input and output data used for training or validation
        trainingData;
        validationData;
        
        % Number of inputs, outputs, and hidden neurons
        numInputs;
        numOutputs;       
        numHidden;
        numSamplesTraining;
        numSamplesValidation;
        
        % The final values from training/validation
        weights;
        
        totalTrainingError; % last epoch only
        totalValidationError;
        
        meanTrainingError;  % last epoch only
        meanValidationError;
        
        % Not available in batch mode
        lastSampleTrainingError;
        lastSampleValidationError;
              
        % x - Each is a sample (only 1 for batch mode)
        % y - Each is an epoch
        % z - Each is an output neuron        
        trainingOutputHistory;                
        
        % x - sample
        % y - epoch
        trainingErrorHistory;
        
        % Record of calculated validation outputs.
        % x - sample
        % y - output
        validationOutputHistory;
        
        % x - output error
        validationErrorHistory;        
        
        % Five-dimensional matrices, first three dimensions are an instance
        % of a weight matrix, fourth dimension is the epoch index, fifth
        % dimension is the sample index (always 1 for batch mode).
        weightHistory;    
        derivativeHistory;    

        hasTrained = 0;        
    end
    
    
    methods
        
        %%
        % Make a matrix for the initial weights of a neural network with
        % the given amount of neurons at each level. The weight matrix has
        % three dimensions/arguments:
        %   1st: Hidden neuron index, or '1' if there isnt any (output biases)
        %   2nd: Input or output neuron index, or '1' if there isn't any (hidden biases)
        %   3rd: "Layer" or group which this type of weight belongs
        %        (input-hidden, hidden bias, hidden-output, output bias).
        %
        %   e.g.    weights(hidden, input/output, group);
        %
        function weights = makeWeightMatrix(this)
            x = this.numHidden;
            y = max(this.numInputs, this.numOutputs);
            z = 4;
            
            weights = zeros(x, y, z);
            
            switch this.weightType
                case NeuralNetwork.WEIGHT_TYPE_I_PLUS_J            
                    for jHidden = 1 : this.numHidden

                        % Input-Hidden link default weights (layer 1)
                        % Values for `c` in RBF
                        for iInput = 1 : this.numInputs
                            weights(jHidden, iInput, 1) = (-1)^(iInput + jHidden);
                        end
                        
                        % Layer 2
                        switch this.modelType
                            % Hidden neuron bias weights
                            case NeuralNetwork.MODEL_TYPE_MLP3                                
                                weights(jHidden, 1, 2) = 1;
                            % Lambda weights
                            case NeuralNetwork.MODEL_TYPE_RBF
                                weights(jHidden, :, 2) = weights(jHidden, :, 1);
                        end
                            
                    end

                    for jHidden = 1 : this.numHidden
                       for kOutput = 1 : this.numOutputs

                           % Hidden-Output link default weights (layer 3)
                            weights(jHidden, kOutput, 3) = (-1)^(jHidden + kOutput);

                            % Output bias weights (layer 4)
                            if jHidden == 1
                                weights(1, kOutput, 4) = 1;
                            end                    
                       end               
                    end    
                case NeuralNetwork.WEIGHT_TYPE_RANDOM
                    for jHidden = 1 : this.numHidden

                        % Input-Hidden link default weights (layer 1)
                        % Values for `c` in RBF
                        for iInput = 1 : this.numInputs
                            weights(jHidden, iInput, 1) = (rand - 0.5) * 2;
                            
                            if this.modelType == NeuralNetwork.MODEL_TYPE_RBF
                                % Lambda weights (layer 2)
                                weights(jHidden, iInput, 2) = (rand - 0.5) * 2;
                            end
                        end
                                                
                        if this.modelType == NeuralNetwork.MODEL_TYPE_MLP3
                            % Hidden neuron biases (layer 2)
                            weights(jHidden, 1, 2) = (rand - 0.5) * 2;
                        end
                    end

                    for jHidden = 1 : this.numHidden
                       for kOutput = 1 : this.numOutputs

                           % Hidden-Output link default weights (layer 3)
                            weights(jHidden, kOutput, 3) = (rand - 0.5) * 2;

                            % Output bias weights (layer 4)
                            if jHidden == 1
                                weights(1, kOutput, 4) = (rand - 0.5) * 2;
                            end                    
                       end               
                    end
            end
        end
        
        %%
        % train()   - Train this neural network using the back-propagation algorithm.
        %
        % Required parameters:
        %   data  - m by n matrix; m = samples, n = inputs and outputs
        %           (input columns and then output columns).
        %   numInputs - the number of inputs in the data table.      
        % Optional parameters:
        %   numHidden - Number of hidden neurons. Default is the mean number
        %             of inputs and output neurons rounded up (ceiling). 
        %   weights - The initial weights to use, given as a three dimensional
        %             matrix, x-y-z, see makeWeightMatrix() for a
        %             description of its format.       
        function epochMeanOutputError = train(this, data, numInputs, ...
                                                     numHidden, initialWeights)
            
            % ===================================
            % Validate input parameters/arguments
            % and initialize state.
            % ===================================                        
            
            % data
            if nargin < 2 || isempty(data) 
                error('train(): Cannot train without data.');
            end
            
            % numInputs
            if nargin < 3 || numInputs < 1
               error('train(): There must be at least one input.');                           
            end      

            % numOutputs
            numOutputs = size(data, 2) - numInputs; %#ok<*PROP>
            if numOutputs < 1
               error('train(): There must be at least one output.'); 
            end

            % numSamples
            numSamples = size(data, 1);
            if numSamples < 1
                error('train(): There must be at least one sample.');
            end

            % inputs
            inputs = data(:, 1:numInputs);

            % outputs
            outputs = data(:, (numInputs + 1):(numInputs + numOutputs));
            
            % numHidden
            if nargin >= 4                
                if numHidden < 1
                    error('train(): There must be at least one hidden neuron.');                 
                end
            else
                % Default number of hidden neurons is the average number of 
                % inputs and outputs, rounded up.
                numHidden = ceil(mean([numInputs numOutputs]));
            end
            
            
            
            % ==========================================================
            % Parameters have been validated. Store relevant information 
            % in the class variables and clear any old data.
            % ==========================================================
            
            this.trainingData              = data;
            this.numInputs                 = numInputs;
            this.numOutputs                = numOutputs;
            this.numHidden                 = numHidden; 
            this.numSamplesTraining        = numSamples;
                        
            % initialWeights
            weights = this.makeWeightMatrix();  
            if nargin >= 5 
                if size(initialWeights) ~= size(weights)
                    error('train(): Size of weight matrix should be %dx%d.', ...
                            size(weights, 1), size(weights, 2));
                else
                    weights = initialWeights;
                end               
            end            
            
            this.weights                   = weights;                        
                        
            this.validationData            = null(1);            
            this.trainingOutputHistory     = null(1);
            this.trainingErrorHistory      = null(1);
            
            this.validationOutputHistory   = null(1);
            this.validationErrorHistory    = null(1);
            
            this.weightHistory             = zeros(size(weights));
            this.derivativeHistory         = zeros(size(weights));

            this.totalTrainingError        = 0; 
            this.totalValidationError      = 0;

            this.meanTrainingError         = 0;  
            this.meanValidationError       = 0;

            this.lastSampleTrainingError   = 0;
            this.lastSampleValidationError = 0;
                        
            this.hasTrained                = 0;
            
            
            % ===========================================================
            % ===========================================================
            %
            % Train using the specified mode and limits.
            %
            % ===========================================================
            % ===========================================================
            
            epochTotalOutputError = Inf;
            
            iEpoch = 1;
            outputError = Inf;
            switch this.trainingMode
                case NeuralNetwork.TRAINING_MODE_SAMPLE_BY_SAMPLE
                    while ~(this.isComplete(iEpoch, epochTotalOutputError))
                        epochTotalOutputError = 0;              
                        for jSample = 1 : numSamples
                            sampleInputs = inputs(jSample, :);
                            sampleOutputs = outputs(jSample, :);                                                                                    
                            
                            % =================
                            % Main part of loop 
                            % =================

                            % Update the weight values (except the very
                            % first occurance, because initial weights have
                            % already been given).
                            if ~(iEpoch == 1 && jSample == 1)
                                this.updateWeights(derivatives);
                            end
                            
                            % Calculate ynn (also return z, so it doesn't
                            % have to be calculated again for the
                            % derivatives).
                            [computedOutputs z] = this.computeOutputs(sampleInputs);
                            
                            % Calculate the error of the outputs in this sample
                            outputError = this.computeError(computedOutputs, sampleOutputs);                            
                                                                                    
                            % Compute the error derivatives for each weight
                            derivatives = this.computeDerivatives(computedOutputs, ...
                                sampleInputs, sampleOutputs, z);
                            
                            
                            epochTotalOutputError = epochTotalOutputError + outputError;
                            
                            % ==========================
                            % Record desired information
                            % ==========================

                            if bitand(this.histories, NeuralNetwork.HISTORY_TYPE_TRAINING_OUTPUTS)
                                this.trainingOutputHistory(jSample, iEpoch, :) = computedOutputs;                                
                            end

                            if bitand(this.histories, NeuralNetwork.HISTORY_TYPE_TRAINING_ERRORS)                                
                                this.trainingErrorHistory(jSample, iEpoch) = outputError;                                
                            end

                            if bitand(this.histories, NeuralNetwork.HISTORY_TYPE_WEIGHTS)
                                this.weightHistory(:, :, :, iEpoch, jSample) = weights;
                            end
                            
                            if bitand(this.histories, NeuralNetwork.HISTORY_TYPE_DERIVATIVES)
                                this.derivativeHistory(:, :, :, iEpoch, jSample) = derivatives;
                            end
                        end
                        iEpoch = iEpoch + 1;
                    end       
                case NeuralNetwork.TRAINING_MODE_BATCH
                    error('train(): Batch mode is not implemented yet.');     
            end
            
            epochMeanOutputError = epochTotalOutputError / numSamples;           
            
            this.totalTrainingError        = epochTotalOutputError;
            this.meanTrainingError         = epochMeanOutputError;  
            this.lastSampleTrainingError   = outputError;            
                        
            this.hasTrained = 1;
        end
        
        %%
        % Perform the validation step on data by using the 
        % weights calculated from training.
        function meanValidationError = validate(this, data)
            
            % ========================================
            % Check parameters and initialize state
            % ========================================
            
            if ~(this.hasTrained)
               error('validate(): Cannot validate without training first.'); 
            end
            
            if nargin < 2 || isempty(data)
                error('validate(): Cannot validate without data.');
            end
            
            if size(data, 2) ~= size(this.trainingData, 2)
               error('validate(): Validation data must have the same number of inputs/outputs as the training data.');
            end
                       
            inputs = data(:, 1:this.numInputs);
            outputs = data(:, (this.numInputs + 1):(this.numInputs + this.numOutputs));
            this.numSamplesValidation = size(data, 1);

            % =======================
            % Main validation routine
            % =======================
            
            totalError = 0;            
            outputError = 0;   
            switch this.trainingMode
                case NeuralNetwork.TRAINING_MODE_SAMPLE_BY_SAMPLE
                    for iSample = 1 : this.numSamplesValidation
                        sampleInputs = inputs(iSample, :);
                        sampleOutputs = outputs(iSample, :);                 
                        
                        computedOutputs = this.computeOutputs(sampleInputs);
                        outputError = this.computeError(computedOutputs, sampleOutputs); 
                        
                        totalError = totalError + outputError;
                        
                        % ==========================
                        % Record desired information
                        % ==========================
                            
                        if bitand(this.histories, NeuralNetwork.HISTORY_TYPE_VALIDATION_OUTPUTS)
                            this.validationOutputHistory(iSample, :) = computedOutputs;                                
                        end
                        
                        if bitand(this.histories, NeuralNetwork.HISTORY_TYPE_VALIDATION_ERRORS)                                
                            this.validationErrorHistory(iSample) = outputError;
                        end
                    end
                case NeuralNetwork.TRAINING_MODE_BATCH
                    error('validate(): Batch mode is not implemented yet.');
            end
            
            meanValidationError            = totalError / this.numSamplesValidation;
            
            this.totalValidationError      = totalError;
            this.meanValidationError       = meanValidationError;
            this.lastSampleValidationError = outputError;
        end
        
        %%
        % Plot the training and validation errors, if they are available.
        function plot(this)            
            % Create a new window
            figure('name', strcat(num2str(this.numInputs), ' Input, ', ...
                            num2str(this.numOutputs), ' Output Neural Net using ', ...
                            num2str(this.numHidden), ' Hidden Neurons'));
            
            switch this.trainingMode
                case NeuralNetwork.TRAINING_MODE_SAMPLE_BY_SAMPLE                      
                    % Training plot (mean and std dev of errors in each epoch)
                    subplot(2, 1, 1), ...
                            errorbar(mean(this.trainingErrorHistory, 1), ...
                            std(this.trainingErrorHistory, 0, 1), ':');
                    title(strcat('Training Errors (sample-by-sample, ', ...
                            num2str(this.numSamplesTraining), ' samples, last epoch mean = ', ...
                            num2str(this.meanTrainingError), ')'));
                    xlabel('Epoch');
                    ylabel('Mean and Std. Dev.');
                    axis([0, Inf, -Inf, Inf]);
                    hold on                    
                        plot(mean(this.trainingErrorHistory, 1), 'r');
                    hold off
                    
                    % Validation plot (shows error of each sample and mean error)       
                    x = linspace(1, this.numSamplesValidation, this.numSamplesValidation);
                    subplot(2, 1, 2), scatter(x, this.validationErrorHistory, 30, '.');                  
                    title(strcat('Validation Errors (', ...
                        num2str(this.numSamplesValidation), ' samples, mean = ', ...
                            num2str(this.meanValidationError), ')'));
                    xlabel('Sample');
                    ylabel('Errors and Mean');
                    axis([0, Inf, -Inf, Inf]);
                    hold on
                        avg = mean(this.validationErrorHistory);
                        y(1:this.numSamplesValidation) = avg;
                        plot(y, 'r');
                    hold off

                case NeuralNetwork.TRAINING_MODE_BATCH
                    error('plot(): Batch mode is not implemented yet.');
            end                            
        end
    end
    
    %%
    methods (Access = private)                
         
        %%
        function result = isComplete(this, iEpoch, currentError)
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
        function [y z] = computeOutputs(this, inputs)                            
            gamma = zeros(this.numHidden, 1);
            z     = zeros(this.numHidden, 1);  
            y     = zeros(this.numOutputs, 1);
            
            for kOutput = 1 : this.numOutputs
                
                % Initial output value is its bias (layer 4)               
                y(kOutput) = this.weights(1, kOutput, 4);
                
                for jHidden = 1 : this.numHidden                    
                    switch this.modelType
                        case NeuralNetwork.MODEL_TYPE_MLP3
                            % Initial gamma value is the hidden neuron's bias
                            gamma(jHidden) = this.weights(jHidden, 1, 2);

                            % Add each term together for gamma
                            for iInput = 1 : this.numInputs                        
                                gamma(jHidden) = gamma(jHidden) + this.weights(jHidden, iInput, 1) * inputs(iInput);
                            end
                            
                            % Calculate z for this hidden neuron
                            z(jHidden) = 1 / (1 + exp(-gamma(jHidden)));
                            
                        case NeuralNetwork.MODEL_TYPE_RBF
                            sum = 0;
                            for iInput = 1 : this.numInputs                        
                                c = this.weights(jHidden, iInput, 1);
                                lambda = this.weights(jHidden, iInput, 2);
                                sum = sum + ((inputs(iInput) - c) / lambda) ^ 2;
                            end
                            gamma(jHidden) = sqrt(sum);
                            
                            % Calculate z for this hidden neuron
                            z(jHidden) = exp(-gamma(jHidden)) ^ 2;
                    end
                    
                    % Add this hidden neuron's effect on the current output
                    y(kOutput) = y(kOutput) + this.weights(jHidden, kOutput, 3) * z(jHidden);
                end          
            end
        end
        
        %% 
        % Compute the error of one sample
        function err = computeError(this, computedOutputs, sampleOutputs)
            err = 0;
            for iOutput = 1 : this.numOutputs
               err = err + .5 * (computedOutputs(iOutput) - sampleOutputs(iOutput))^2;
            end
        end
        
        %%
        % Derivative of the error with respect to each weight. The
        % derivatives matrix has the same format as the weight matrix (i.e.
        % four layers/groups).
        function derivatives = computeDerivatives(this, computedOutputs, ...
                sampleInputs, sampleOutputs, z)
            
            derivatives = zeros(size(this.weights));
            
            for kOutput = 1 : this.numOutputs
                
                % Derivative with respect to the output
                % ynn - ytable
                derivatives(1, kOutput, 4) = computedOutputs(kOutput) - sampleOutputs(kOutput);
                
                for jHidden = 1 : this.numHidden;
                    
                    % Derivative with respect to hidden-output link weights
                    % dE/dy * zj
                    derivatives(jHidden, kOutput, 3) = derivatives(1, kOutput, 4) * z(jHidden);
                    
                    switch this.modelType
                        case NeuralNetwork.MODEL_TYPE_MLP3
                            % Derivative with respect to hidden neuron biases
                            % sum(dE/dy * (hidden-output link derivative) * zj * (1 - zj))
                            derivatives(jHidden, 1, 2) = derivatives(jHidden, 1, 2) ...
                                + derivatives(1, kOutput, 4) ...
                                * derivatives(jHidden, kOutput, 3) ...
                                * z(jHidden) * (1 - z(jHidden));

                            % Derivative with respect to input-hidden link weights
                            % sum(dE/dy * (hidden-output link derivative) * zj * (1 - zj) * xi)
                            for iInput = 1 : this.numInputs
                                derivatives(jHidden, iInput, 1) = derivatives(jHidden, iInput, 1) ...
                                    + derivatives(1, kOutput, 4) ...
                                    * derivatives(jHidden, kOutput, 3) ...
                                    * z(jHidden) * (1 - z(jHidden)) * sampleInputs(iInput);                                
                            end
                        case NeuralNetwork.MODEL_TYPE_RBF
                            % nothing
                    end
                end
            end
            
            if this.modelType == NeuralNetwork.MODEL_TYPE_RBF
               for jHidden = 1 : this.numHidden;
                   sum = 0;
                   for kOutput = 1 : this.numOutputs
                      sum = sum ...
                          + derivatives(1, kOutput, 4) ...
                          * this.weights(jHidden, kOutput, 3);
                   end
                   
                   for iInput = 1 : this.numInputs
                       %lambda
                       derivatives(jHidden, iInput, 2) = sum * 2 * z(jHidden) ...
                           * (sampleInputs(iInput) - this.weights(jHidden, iInput, 1)) ^ 2 ...
                           / this.weights(jHidden, iInput, 2) ^ 3;
                       
                       %c
                       derivatives(jHidden, iInput, 1) = sum * 2 * z(jHidden) ...
                           * (sampleInputs(iInput) - this.weights(jHidden, iInput, 1)) ...
                           / this.weights(jHidden, iInput, 2) ^ 2;
                   end
               end
            end
        end
        
        %%
        % Update weights
        function updateWeights(this, derivatives)                      
                 
            % Update input-hidden link weights (layer 1)
            for iInput = 1 : this.numInputs                
                for jHidden = 1 : this.numHidden
                    this.weights(jHidden, iInput, 1) = this.weights(jHidden, iInput, 1) ...
                        - derivatives(jHidden, iInput, 1) * this.alpha;
                end
            end
            
            
            for jHidden = 1 : this.numHidden
                
                % Update hidden neuron biases (layer 2)
                this.weights(jHidden, 1, 2) = this.weights(jHidden, 1, 2) ...
                    - derivatives(jHidden, 1, 2) * this.alpha;

                % Update hidden-output link weights (layer 3)
                for kOutput = 1 : this.numOutputs                        
                    this.weights(jHidden, kOutput, 3) = this.weights(jHidden, kOutput, 3) ...
                        - derivatives(jHidden, kOutput, 3) * this.alpha;                    
                end
            end
            
             % Update output neuron biases (layer 4)
            for kOutput = 1 : this.numOutputs               
                this.weights(1, kOutput, 4) = this.weights(1, kOutput, 4) ...
                    - derivatives(1, kOutput, 4) * this.alpha;  
            end
            
        end
    end
end