classdef NeuralNetwork < handle
    
    %% Constants used instead of the corresponding numbers to help ease reading the code.
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
       HISTORY_TYPE_NONE               = bin2dec('0000000');
       HISTORY_TYPE_TRAINING_OUTPUTS   = bin2dec('0000001');
       HISTORY_TYPE_TRAINING_ERRORS    = bin2dec('0000010');
       HISTORY_TYPE_VALIDATION_OUTPUTS = bin2dec('0000100');
       HISTORY_TYPE_VALIDATION_ERRORS  = bin2dec('0001000');
       HISTORY_TYPE_WEIGHTS            = bin2dec('0010000');
       HISTORY_TYPE_DERIVATIVES        = bin2dec('0100000');       
       HISTORY_TYPE_ALPHAS             = bin2dec('1000000');
       % Combination histories
       HISTORY_TYPE_TRAINING           = bin2dec('0000011');
       HISTORY_TYPE_VALIDATION         = bin2dec('0001100');
       HISTORY_TYPE_OUTPUTS            = bin2dec('0000101');
       HISTORY_TYPE_ERRORS             = bin2dec('0001010');
       HISTORY_TYPE_ALL                = bin2dec('1111111');
       
       INITIAL_WEIGHTS_ALTERNATING     = 1;  % e.g. (-1)^(i+j)
       INITIAL_WEIGHTS_CONSTANT        = 2;
       INITIAL_WEIGHTS_RANDOM          = 3;
    end

    %% User definable properties
    properties
        modelType           = NeuralNetwork.MODEL_TYPE_MLP3;
        initialWeightType   = NeuralNetwork.INITIAL_WEIGHTS_ALTERNATING;
        trainingMode        = NeuralNetwork.TRAINING_MODE_SAMPLE_BY_SAMPLE;
        terminationMode     = NeuralNetwork.TERMINATION_MODE_EITHER;  
        histories           = NeuralNetwork.HISTORY_TYPE_ERRORS + NeuralNetwork.HISTORY_TYPE_ALPHAS;
        
        maxEpochs           = 40;
        maxError            = 0.001;
        
        alphaConstant       = 0.1;
        
        useVariableAlpha    = 1;       
        alphaMin            = 0;
        alphaMax            = 1.0;
        epsilon             = 0.01;
        
    end
    
    %% User readable properties, usually read after training/validation.
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
        
        l1Error; % L1
        l2Error; % L2
        
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
        
        % x - sample
        % y - epoch
        alphaHistory;

        hasTrained = 0;        
    end
    
    %% Static methods not dependent on a specific NeuralNetwork instance.
    methods (Static)
        function weightSize = getWeightSize(numInputs, numHidden, numOutputs)
           weightSize = [numHidden, max(numInputs, numOutputs), 4]; 
        end
    end
    
    %% Public methods accessible by the user.
    methods        
        %% train() - Train this neural network using the back-propagation algorithm.
        %
        % Required parameters:
        %   
        %   data  - m by n matrix; m = samples, n = inputs and outputs
        %           (input columns first).       
        %
        % Optional parameters:
        %   
        %   numOutputs - The number of outputs, default = 1.
        %
        %   numHidden - Number of hidden neurons. Default is the mean number
        %             of inputs and output neurons rounded up (ceiling). 
        %   weights - The initial weights to use, given as either a scalar 
        %             used as a default value, or a three dimensional
        %             matrix, x-y-z, see makeWeightMatrix() for a
        %             description of its format.       
        function epochMeanOutputError = train(this, data, numOutputs, ...
                                                     numHidden, initialWeights)
            
            % ===================================
            % Validate input parameters/arguments
            % and initialize state.
            % ===================================                        
            
            if nargin < 2 || isempty(data) 
                error('train(): Cannot train without data.');
            end

            if nargin >= 3
                if numOutputs < 1
                    error('train(): There must be at least one output.');  
                end
            else
                numOutputs = 1;
            end      

            numInputs = size(data, 2) - numOutputs; %#ok<*PROP>
            if numInputs < 1
               error('train(): There must be at least one input.'); 
            end

            numSamples = size(data, 1);
            if numSamples < 1
                error('train(): There must be at least one sample.');
            end

            inputs = data(:, 1:numInputs);

            outputs = data(:, (numInputs + 1):(numInputs + numOutputs));
            
            if nargin >= 4                
                if numHidden < 1
                    error('train(): There must be at least one hidden neuron.');                 
                end
            else
                numHidden = ceil(mean([numInputs numOutputs]));
            end
            
            if nargin >= 5
                if max(size(initialWeights)) == 1 
                    weights = this.makeWeightMatrix(numInputs, numHidden, ...
                            numOutputs, initialWeights);
                else
                    weightSize = NeuralNetwork.getWeightSize( ...
                            numInputs, numOutputs, numHidden);
                    if size(initialWeights) ~= weightSize
                        error('train(): Size of weight matrix should be %dx%dx4.', ...
                                weightSize(1), weightSize(2));   
                    else
                        weights = initialWeights;
                    end  
                end
            else
                weights = this.makeWeightMatrix(numInputs, numHidden, numOutputs);
            end
        
            
            % =================================================
            % Store relevant information in the class variables 
            % and clear any old data.
            % =================================================
            
            this.trainingData              = data;
            this.numInputs                 = numInputs;
            this.numOutputs                = numOutputs;
            this.numHidden                 = numHidden; 
            this.numSamplesTraining        = numSamples;                                                           
            
            this.weights                   = weights;                        
                        
            this.validationData            = null(1);            
            this.trainingOutputHistory     = null(1);
            this.trainingErrorHistory      = null(1);
            
            this.validationOutputHistory   = null(1);
            this.validationErrorHistory    = null(1);            
            
            this.weightHistory             = zeros(size(weights));
            this.derivativeHistory         = zeros(size(weights));
            
            this.alphaHistory              = null(1);

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
            
            
            
            iEpoch = 1;
            epochTotalOutputError = Inf;                                              
            alpha = this.alphaConstant;            
            switch this.trainingMode
                case NeuralNetwork.TRAINING_MODE_SAMPLE_BY_SAMPLE
                    while ~(this.isComplete(iEpoch, epochTotalOutputError))
                        
                        epochTotalOutputError = 0;
                        for jSample = 1 : numSamples
                            x = inputs(jSample, :);
                            Ydata = outputs(jSample, :);                                                                                    
                            
                            % =================
                            % Main part of loop 
                            % =================

                            % Update the weight values (except the very
                            % first occurance, because initial weights have
                            % already been given).
                            if ~(iEpoch == 1 && jSample == 1)
                                if this.useVariableAlpha
                                    alpha = this.calcAlpha(x, Ydata, weights, derivs); 
                                end
                                weights = this.updateWeights(weights, derivs, alpha);
                            end
                            
                            % Calculate ynn (also return z, so it doesn't
                            % have to be calculated again for the derivs).                            
                            [Ynn z] = this.calcYnn(x, weights);
                            
                            % Calculate the error of the outputs in this sample                            
                            E = this.calcError(Ynn, Ydata);                            
                                                                                    
                            % Calculate the error derivs for each weight
                            % i.e. [a b c d] = calcDerivs(x, Ynn, Ydata, z)
                            derivs = this.calcDerivs(x, Ynn, Ydata, weights, z);                                                        
                                                        
                            epochTotalOutputError = epochTotalOutputError + E;
                            
                            % ==========================
                            % Record desired information
                            % ==========================

                            if bitand(this.histories, NeuralNetwork.HISTORY_TYPE_TRAINING_OUTPUTS)
                                this.trainingOutputHistory(jSample, iEpoch, :) = Ynn;                                
                            end

                            if bitand(this.histories, NeuralNetwork.HISTORY_TYPE_TRAINING_ERRORS)                                
                                this.trainingErrorHistory(jSample, iEpoch) = E;                                
                            end

                            if bitand(this.histories, NeuralNetwork.HISTORY_TYPE_WEIGHTS)
                                this.weightHistory(:, :, :, iEpoch, jSample) = weights;
                            end
                            
                            if bitand(this.histories, NeuralNetwork.HISTORY_TYPE_DERIVATIVES)
                                this.derivativeHistory(:, :, :, iEpoch, jSample) = derivs;
                            end
                            
                            if bitand(this.histories, NeuralNetwork.HISTORY_TYPE_ALPHAS)
                                this.alphaHistory(jSample, iEpoch) = alpha;
                            end
                        end
                        iEpoch = iEpoch + 1;
                    end       
                case NeuralNetwork.TRAINING_MODE_BATCH
                    error('train(): Batch mode is not implemented yet.');     
            end
            
            this.weights = weights;
            
            epochMeanOutputError = epochTotalOutputError / numSamples;           
            
            this.totalTrainingError        = epochTotalOutputError;
            this.meanTrainingError         = epochMeanOutputError;  
            this.lastSampleTrainingError   = E;            
                        
            this.hasTrained = 1;
        end
        
        %% validate() - Perform the validation step on data by using the weights calculated from training.
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
            totalL1Error = 0;
            totalL2Error = 0;
            E = 0;   
            switch this.trainingMode
                case NeuralNetwork.TRAINING_MODE_SAMPLE_BY_SAMPLE
                    for iSample = 1 : this.numSamplesValidation
                        x = inputs(iSample, :);
                        Ydata = outputs(iSample, :);                 
                                                
                        Ynn = this.calcYnn(x, this.weights);                                                
                        E = this.calcError(Ynn, Ydata); 
                        
                        totalError = totalError + E;
                        totalL1Error = totalL1Error + this.calcL1Error(Ynn, Ydata); 
                        totalL2Error = totalL2Error + this.calcL2Error(Ynn, Ydata); 
                        
                        % ==========================
                        % Record desired information
                        % ==========================
                            
                        if bitand(this.histories, NeuralNetwork.HISTORY_TYPE_VALIDATION_OUTPUTS)
                            this.validationOutputHistory(iSample, :) = Ynn;                                
                        end
                        
                        if bitand(this.histories, NeuralNetwork.HISTORY_TYPE_VALIDATION_ERRORS)                                
                            this.validationErrorHistory(iSample) = E;
                        end
                    end
                case NeuralNetwork.TRAINING_MODE_BATCH
                    error('validate(): Batch mode is not implemented yet.');
            end
            
            meanValidationError            = totalError / this.numSamplesValidation;            
            this.l1Error                   = totalL1Error / this.numSamplesValidation;
            this.l2Error                   = sqrt(totalL2Error) / this.numSamplesValidation;
            
            this.totalValidationError      = totalError;
            this.meanValidationError       = meanValidationError;
            this.lastSampleValidationError = E;
        end
        
        %% plot() - Plot the training and validation errors, if they are available.
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
    
    %% Private methods, only accessible by instances of this class (e.g. 'nn');
    methods (Access = private)     
        
        %% makeWeightMatrix()
        % Create a matrix for the initial weights of a neural network with
        % the given amount of neurons at each level. The weight matrix has
        % three dimensions/arguments:
        %   1st: Hidden neuron index, or '1' if there isnt any (output biases)
        %   2nd: Input or output neuron index, or '1' if there isn't any (hidden biases)
        %   3rd: "Layer" or group which this type of weight belongs
        %        (input-hidden, hidden bias, hidden-output, output bias).
        %
        %   e.g.    weights(hidden, input/output, group, numAlphas);
        %
        function weights = makeWeightMatrix(this, numInputs, numHidden, numOutputs, defaultWeight)            
            if nargin < 3
                defaultWeight = 1;
            end
                        
            weights = zeros(numHidden, max(numInputs, numOutputs), 4);
            
            switch this.initialWeightType
                case NeuralNetwork.INITIAL_WEIGHTS_ALTERNATING            
                    for j = 1 : numHidden

                        % Input-Hidden weights (layer 1, or "a")
                        % (Values for "Cij" in RBF)
                        for i = 1 : numInputs
                            weights(j, i, 1) = (-1)^(i + j);                            
                        end
                        
                        % Layer 2, or "b"
                        if this.modelType == NeuralNetwork.MODEL_TYPE_RBF                                                            
                            % Lambda weights (same as Cij in this case)
                            weights(j, :, 2) = weights(j, :, 1);
                        end
                            
                    end
                    
                    if this.modelType == NeuralNetwork.MODEL_TYPE_MLP3
                        % Hidden bias weights
                        weights(:, 1, 2) = defaultWeight;
                    end
                        
                    % Hidden-Output weights (layer 3, or "c")
                    for j = 1 : numHidden
                       for k = 1 : numOutputs
                            weights(j, k, 3) = (-1)^(j + k);                
                       end               
                    end
                    
                    % Output bias weights (layer 4, or "d")                    
                    weights(1, :, 4) = defaultWeight;                                    
                    
                    
                case NeuralNetwork.INITIAL_WEIGHTS_RANDOM
                    for j = 1 : numHidden

                        % Input-Hidden link default weights (layer 1, or "a")
                        % (Values for "Cij" in RBF)
                        for i = 1 : numInputs
                            weights(j, i, 1) = (rand - 0.5) * 2;
                            
                            if this.modelType == NeuralNetwork.MODEL_TYPE_RBF
                                % Lambda weights (layer 2, or "b")
                                weights(j, i, 2) = (rand - 0.5) * 2;
                            end
                        end
                                                
                        if this.modelType == NeuralNetwork.MODEL_TYPE_MLP3
                            % Hidden biases (layer 2, or "b")
                            weights(j, 1, 2) = (rand - 0.5) * 2;
                        end
                    end

                    % Hidden-Output weights (layer 3, or "c")
                    for j = 1 : numHidden
                       for k = 1 : numOutputs
                            weights(j, k, 3) = (rand - 0.5) * 2;                
                       end               
                    end
                    
                    % Output bias weights (layer 4, or "d")
                    for k = 1 : numOutputs
                        weights(1, k, 4) = (rand - 0.5) * 2;                                    
                    end
                    
                case NeuralNetwork.INITIAL_WEIGHTS_CONSTANT
                    weights(:, :, :) = defaultWeight;
                    
            end
        end
         
        %% isComplete() - Check if stopping conditions have been met.
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
        
        %% calcYnn() - Compute output(s) of one sample
        function [Ynn z] = calcYnn(this, x, weights)                            
            gamma = zeros(this.numHidden, 1);
            z     = zeros(this.numHidden, 1);  
            Ynn     = zeros(this.numOutputs, 1);
            
            switch this.modelType
                case NeuralNetwork.MODEL_TYPE_MLP3
                    for k = 1 : this.numOutputs
                        V0k = weights(1, k, 4);
                        Ynn(k) = V0k; % Initial value
                        for j = 1 : this.numHidden
                            U0j = weights(j, 1, 2);
                            gamma(j) = U0j; % Initial value
                            for i = 1 : this.numInputs                        
                                Uij = weights(j, i, 1);
                                gamma(j) = gamma(j) + Uij * x(i);
                            end                                                        
                            z(j) = 1 / (1 + exp(-gamma(j)));
                        end                        
                        Vjk = weights(j, k, 3);
                        Ynn(k) = Ynn(k) + Vjk * z(j);
                    end
                            
                case NeuralNetwork.MODEL_TYPE_RBF
                    for k = 1 : this.numOutputs             
                        V0k = weights(1, k, 4);
                        Ynn(k) = V0k; % Initial value
                        for j = 1 : this.numHidden
                            sum = 0;
                            for i = 1 : this.numInputs                        
                                Cij = weights(j, i, 1);
                                LAMBDAij = weights(j, i, 2);
                                sum = sum + ((x(i) - Cij) / LAMBDAij) ^ 2;
                            end
                            gamma(j) = sqrt(sum);
                            z(j) = exp(-gamma(j)) ^ 2;               
                        end                        
                        Vjk = weights(j, k, 3);
                        Ynn(k) = Ynn(k) + Vjk * z(j);
                    end  
            end
        end
        
        %% calcError() - Compute the error of one sample
        function E = calcError(this, Ynn, Ydata) 
            E = 0;
            for k = 1 : this.numOutputs
               E = E + .5 * (Ynn(k) - Ydata(k)) ^ 2;
            end
        end
        
        %% calcL1Error() - Compute the L1 error of one sample
        function E = calcL1Error(this, Ynn, Ydata)     
            E = 0;
            for k = 1 : this.numOutputs
               E = E + abs(Ynn(k) - Ydata(k));
            end
        end

        %% calcL2Error() - Compute the L2 error of one sample
        function E = calcL2Error(this, Ynn, Ydata)
            E = 0;
            for k = 1 : this.numOutputs
               E = E + (Ynn(k) - Ydata(k)) ^ 2;
            end
        end
        
        %% calcDerivs()
        % Derivative of the error with respect to each weight. The
        % derivs matrix has the same format as the weight matrix (i.e.
        % four layers/groups).
        function derivs = calcDerivs(this, x, Ynn, Ydata, weights, z)
            
            derivs = zeros(size(weights));
            % Derivative groups:            
            % a = MLP3 - derivs of input-hidden layer weights
            %     RBF  - derivs of "C" weights
            % b = MLP3 - derivs of hidden layer biases
            %     RBF  - derivs of "lambda" weights
            % c = derivs of hidden-output layer weights
            % d = derivs of output layer biases
            
            switch this.modelType
                case NeuralNetwork.MODEL_TYPE_MLP3
                    for k = 1 : this.numOutputs
                        
                        % output bias
                        d = Ynn(k) - Ydata(k);

                        for j = 1 : this.numHidden;

                            % hidden-output weight
                            c = d * z(j);

                            % hidden bias                            
                            b = d * c * z(j) * (1 - z(j));
                            
                            for i = 1 : this.numInputs
                                
                                % input-hidden weight
                                a = d * c * z(j) * (1 - z(j)) * x(i);  
                                
                                derivs(j, i, 1) = a;
                            end                            
                            derivs(j, 1, 2) = b;
                            derivs(j, k, 3) = c;
                        end                    
                        derivs(1, k, 4) = d;
                    end
                case NeuralNetwork.MODEL_TYPE_RBF                    
                    for k = 1 : this.numOutputs                        
                        % output bias
                        d = Ynn(k) - Ydata(k);
                        derivs(1, k, 4) = d;                       
                    end
                    
                    % Prepare for lambda and C
                    % summation of: ((ynn,k) - yk) * Vjk
                    sum = zeros(this.numHidden, 1);
                    for j = 1 : this.numHidden;
                        for k = 1 : this.numOutputs
                            Vjk = weights(j, k, 3);
                            sum(j) = sum(j) + d * Vjk;
                        end
                    end 
                    
                    for k = 1 : this.numOutputs
                        for j = 1 : this.numHidden

                            % hidden-output weight
                            c = d * z(j);

                            for i = 1 : this.numInputs
                                
                                % c-weight (not to be confused with derivative group c)
                                Cij = weights(j, i, 1);
                                LAMBDAij = weights(j, i, 2);
                                
                                b = sum(j) * 2 * z(j) * (x(i) - Cij) ^ 2 / LAMBDAij ^ 3;
                                
                                a = sum(j) * 2 * z(j) * (x(i) - Cij) / LAMBDAij ^ 2;
                                
                                derivs(j, i, 1) = a;
                                derivs(j, i, 2) = b;                                
                            end                            
                            derivs(j, k, 3) = c;
                        end
                        
                    end 
            end
        end
        
        %% calcAlpha() - Compute what alpha should be used for this update 
        % TODO: dependent on sample-by-sample method
        function alpha = calcAlpha(this, x, Ydata, weights, derivs)
            
            % Using linear search algorithm to find where alpha converges.
            % The magnitude of alpha values are always in the order
            % a1 ... a3 ... a4 ... a2
            a1 = this.alphaMin;
            a2 = this.alphaMax;                     
            
            a3 = a2 - 0.618 * (a2 - a1);
            a4 = a1 + 0.618 * (a2 - a1);
            
            w3 = this.updateWeights(weights, derivs, a3);
            w4 = this.updateWeights(weights, derivs, a4);

            Ynn3 = this.calcYnn(x, w3);
            Ynn4 = this.calcYnn(x, w4); 

            E3 = this.calcError(Ynn3, Ydata);
            E4 = this.calcError(Ynn4, Ydata);
            
            while 1
                if E3 > E4
                    a1 = a3;
                    a3 = a4;
                    a4 = a1 + 0.618 * (a2 - a1);
                    
                    if (a2 - a1) <= this.epsilon
                        alpha = a3; % or any a#  
                        break;
                    end
                    
                    E3 = E4;
                    w4 = this.updateWeights(weights, derivs, a4);
                    Ynn4 = this.calcYnn(x, w4); 
                    E4 = this.calcError(Ynn4, Ydata);
                else
                    a2 = a4;
                    a4 = a3;
                    a3 = a2 - 0.618 * (a2 - a1);
                    
                    if (a2 - a1) <= this.epsilon
                        alpha = a3; % or any a#  
                        break;
                    end
                    
                    E4 = E3;
                    w3 = this.updateWeights(weights, derivs, a3);
                    Ynn3 = this.calcYnn(x, w3); 
                    E3 = this.calcError(Ynn3, Ydata);
                end
            end
        end
        
        %% updateWeights()
        function weights = updateWeights(this, weights, derivs, alpha)                 
            switch this.modelType
                case NeuralNetwork.MODEL_TYPE_MLP3 
                    for i = 1 : this.numInputs                
                        for j = 1 : this.numHidden                    
                            Uij = weights(j, i, 1);
                            dUij = derivs(j, i, 1); 

                            % input-hidden
                            Uij = Uij - dUij * alpha;

                            weights(j, i, 1) = Uij;
                        end
                    end            

                    for j = 1 : this.numHidden                
                        U0j = weights(j, 1, 2);
                        dU0j = derivs(j, 1, 2);

                        % hidden bias
                        U0j = U0j - dU0j * alpha;

                        for k = 1 : this.numOutputs   
                            Vjk = weights(j, k, 3);
                            dVjk = derivs(j, k, 3);

                            % hidden-output
                            Vjk = Vjk - dVjk * alpha;    

                            weights(j, k, 3) = Vjk;
                        end
                        weights(j, 1, 2) = U0j;
                    end

                    for k = 1 : this.numOutputs 
                        V0k = weights(1, k, 4);
                        dV0k = derivs(1, k, 4);

                        % output bias
                        V0k = V0k - dV0k * alpha;

                        weights(1, k, 4) = V0k;
                    end
                case NeuralNetwork.MODEL_TYPE_RBF
                    for i = 1 : this.numInputs                
                        for j = 1 : this.numHidden                    
                            Cij = weights(j, i, 1);
                            dCij = derivs(j, i, 1); 
                            
                            LAMBDAij = weights(j, i, 2);
                            dLAMBDAij = derivs(j, i, 2);

                            % input-hidden C
                            Cij = Cij - dCij * alpha;
                            
                            % input-hidden LAMBDA
                            LAMBDAij = LAMBDAij - dLAMBDAij * alpha;
                            
                            weights(j, i, 1) = Cij;
                            weights(j, i, 2) = LAMBDAij;
                        end
                    end            

                    for j = 1 : this.numHidden
                        for k = 1 : this.numOutputs   
                            Vjk = weights(j, k, 3);
                            dVjk = derivs(j, k, 3);

                            % hidden-output
                            Vjk = Vjk - dVjk * alpha;    

                            weights(j, k, 3) = Vjk;
                        end
                    end

                    for k = 1 : this.numOutputs 
                        V0k = weights(1, k, 4);
                        dV0k = derivs(1, k, 4);

                        % output bias
                        V0k = V0k - dV0k * alpha;

                        weights(1, k, 4) = V0k;
                    end
            end
        end
    end
end