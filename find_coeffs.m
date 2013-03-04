% Jacob Phillips and Sreeram
% Feb. 12, 2013

function coeffs = find_coeffs(~)
    coeffs = initial_coeffs();
    data = initial_data_points();
    
    inputs = get_inputs(data);
    outputs = get_outputs(data);
    
    outputs_calculated = calc_outputs(inputs, coeffs);
    err = calc_error(outputs, outputs_calculated);

    count = 0;
    while err > .0001  
        coeff_errors = calc_coeff_errors(coeffs, inputs, outputs, outputs_calculated);
        coeffs = adjust_coeffs(coeffs, err, coeff_errors);
        outputs_calculated = calc_outputs(inputs, coeffs);
        err = calc_error(outputs, outputs_calculated);
        count = count + 1;
    end
    
    fprintf('After %d iterations, the error has been reduced to %d.', count, err);
end

% Initial data table (as given in problem).
function data = initial_data_points() 
    data = [-1, -2.5;
            0, 1.5;
            1, 10.64;
            2, 48.5];
end

% Initial guesses for the coefficients (as given in problem).
function coeffs = initial_coeffs()
    coeffs = [1, 2, 3, 4];
end

% Inputs (x values) are the first column of the matrix.
function inputs = get_inputs(data)
    inputs = data(:, 1);
end

% Outputs (y values) are the second column of the matrix.
function outputs = get_outputs(data)
    outputs = data(:, 2);
end

% Calculate the output of each input using the given coefficients.
function outputs = calc_outputs(inputs, coeffs)
    for i = 1 : size(inputs, 1)
        x = inputs(i);
        outputs(i) = coeffs(1) + coeffs(2)*x + coeffs(3)*x^2 + coeffs(4)*x^3;
    end
    % Return column vector
    outputs = transpose(outputs);
end

% Calculate the error by comparing given y values to calulated y values.
function err = calc_error(outputs, outputs_calculated)
    err = 0;
    for i = 1 : size(outputs, 1)
        err = err + ((outputs(i) - outputs_calculated(i))^2)/2;        
    end
end

% Each coefficient's error is calculated as the sum of its errors
% on each individual input/output combination.
function errors = calc_coeff_errors(coeffs, inputs, outputs, outputs_calculated)
    % For each coefficient
    for i = 1 :  size(coeffs, 2)
        errors(i) = 0;
        % Add up all of the errors of this coefficient.        
        for j = 1 : size(inputs, 1)
            errors(i) = errors(i) + (outputs_calculated(j) - outputs(j)) * inputs(j) ^ (i - 1);     
        end
    end
end

% Calculate the adjustments to each coefficient.
function coeffs = adjust_coeffs(coeffs, err, coeff_errors)
    num_coeffs = size(coeff_errors, 2);
    
    % Total magnitude of coefficients' errors
    total = 0;    
    for i = 1 : num_coeffs
        total = total + abs(coeff_errors(i));
    end
    
    
    for i = 1 : num_coeffs;
		% have to take absolute value because (1 - percentage) should always
		% be within [0,1] 
        percentage = abs(coeff_errors(i)) / total;
        
        % num_coeffs * 1.3 yields results faster for e of .0001 (about 4 times 
        % as many iterations for a 10 fold increase in precision). Beyond
        % .0001, the iterations were increasing slightly more than
        % linearly, while num_coeffs^2 always increased slightly less than
        % linearly. However, the total iterations was always less using
        % 1.3. Neither method allows matlab to finish (not crash) to an error 
		% less than 0.0000001.
        sharing_factor = num_coeffs ^ 2;
        
        adj = err * ((1 - percentage) / sharing_factor);
        
        if coeff_errors(i) > 0
            coeffs(i) = coeffs(i) - adj;
        else
            coeffs(i) = coeffs(i) + adj;
        end
    end  
end