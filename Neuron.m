classdef (Abstract) Neuron
    properties (Access = protected)      
        input;      % Scalar entering neuron
        output;     % Scalar leaving neuron
    end
    methods (Abstract)
       feed_forward(~); 
    end
end

