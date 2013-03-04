function data = additional_data_points() 
    data = [-2, f(-2);
            -3, f(-3);
            3, f(3);
            4, f(4)];
end

function y = f(x)    
    y = perturb(1 + 2*x + 3*x^2 + 4*x^3);
end

% Modify the input variable by a small random amount [-1,1].
function x2 = perturb(x)	
    x2 = x + (rand - 0.5) * 2;
end