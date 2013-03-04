classdef InputNeuron < Neuron
    methods
        function obj = InputNeuron(value)
           obj.input = value; 
           obj.output = value;
        end
        
        function obj = feed_forward(~)
            
        end
    end
end