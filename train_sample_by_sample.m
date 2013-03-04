% inputs = Matrix of input values, each row is a set of inputs.
% outputs = Matrix of output values, each row is a set of outputs.
% epochs = number of epochs to train the neural network with, default = 100
% weights = Vector (horizontal) of the various weights, default = 1.0
% num_hidden = number of hidden neurons, default = # of input rows.
function weights = train_sample_by_sample(inputs, outputs, epochs, weights, num_hidden)
    if nargin < 2 || isempty(inputs) || isempty(outputs)
        throw(MException('train_sample_by_sample()', ...
                         'Cannot train without data.'));
    end
    
    if isempty(epochs)
        epochs = 100;
    end

    if isempty(num_hidden)
        num_hidden = size(inputs, 1);
    end
    
    % Vector of the size of each level of weights
    % w(1) = weights between input and hidden layer
    % w(2) = weights between hidden and output layer
    % w(3) = biases of hidden layer
    % w(4) = biases of output layer 
    num_weights = [
        (size(inputs, 2) * num_hidden)
        (num_hidden * size(outputs, 2))
        num_hidden
        size(outputs, 2)
    ];

    if isempty(weights)
        weights = ones(sum(num_weights));
    end           
    

end

function outputs = calc_outputs(inputs, weights, num_weights) 
    
    input_biases = weights(num_weights(3));


    gammas = ones(num_hidden);
    zeds = ones(num_hidden);
    
    
    for i = 1 : num_hidden
        gammas(i) = inputs
    end
end