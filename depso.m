% Terms
% 
% de        Differential Evolution
% pso       Particle Swarm Optimization
% depso     DE and PSO combined
% pop       Population (the swarm)
% x         An individual, particle, position, or weight vector (all mean the same)
% v         Velocity vector
% xr        Random particle
% z         DE mutated particle
% u         DE crossed-over particle
% F         DE mutate scaling factor
% CR        DE crossover probability
% pbest     Particle's best known position
% gbest     Swarm's best known position
% gbeste    Error of best position
% w         PSO inertia weight
% c1        PSO cognitive factor
% c2        PSO social factor
% S         Entropy
% alpha     Entropy decay rate [0,1]
%
% pop_hist      Population history
% err_hist      Error/fitness history
% gbest_hist     Global best weights history
% gbeste_hist    Global best error history


% Input/Output Data Matrix
%
%   x-dimension: Each row is a sample
%   y-dimension: Each column is an input or output variable


% Population Matrix (with history)
%
%   x-dimension: Each row consists of two weights
%   y-dimension: Column 1 = position weights, 
%                Column 2 = velocity weights
%   z-dimension: Each is a particle/individual (has position and velocity)
% 4th-dimension: Generation/epoch index


% Error/Fitness Matrix (with history)
%
%   x-dimension: Each row is a particle's error/fitness value
%   y-dimension: Generation/epoch index


% Global best weights (with history)
%
%   x-dimension: Each row represents weights
%   y-dimension: Each column is gbest for the generation/epoch

% Global best error (with history)
%
%   x-dimension: Each row represents the gbest error for the generation/epoch

%%
classdef Depso < handle
    properties (Constant)
        
    end    
    
    properties % read and write access
        % PSO factors
        w  = 0.7298
        c1 = 1.496
        c2 = 1.496        
        VmaxMin = .2;
        VmaxMax = .8;
        
        % DE factors
       Fmax  = 1;
       Fmin  = 0.5;       
       CRmin = 1;
       CRmax = 1;
       
       % Local factors
       Lmax = .00;
       Lmin = .00;
       

       popsize = 20;          
       lsEpochs = 3;
       lsEpochsExtended = 10;  
       lsDerivThreshold = 0.01; % change when getting random large error means.
       numTrainingSamples;
       numTestSamples;
       numInputs;
       numOutputs;
       numHidden;    
       pop;
       gbest;
       gbeste = inf;
       iGen;   
       
       alpha = 1;
       dist  = 1;    % sort of "distance from goal"
       
       % Termination conditions (any will stop)
       termGen    = 100;
       termErr    = 0.001;
       termConv   = 0.000001; 
       termImp    = 0.0001;   % Total improvement
       termImpGen = 10;      % over x generations 
       termImpS   = 0.1;    % AND entropy is less than y
       imp = ones(1, 10); % same num as termImpGen !!!
    end    
    properties (SetAccess = private) % read-only for user
       S          = 1;       
       pop_hist;
       err_hist;
       gbest_hist;
       gbeste_hist;
       numWeights;
    end
    
    properties (Hidden)                
        iParticle; % used by DE mutate
        dataset;            % training or validation
        
        % used for efficency
        Ex; 
        canUseEx = 0;        
    end
 
    %%
    methods
        %%
        % Initialize population
        %
        function pop = initPop(this, popsize, numInputs, numOutputs, numHidden) 
            if nargin < 2 || popsize < 1;
               popsize = this.popsize;
            end
            if nargin < 3 || numInputs < 1;
               numInputs = 1;
            end
            if nargin < 4 || numOutputs < 1;
               numOutputs = 1;
            end
            if nargin < 5 || numHidden < 1
               numHidden = ceil(mean([numInputs numOutputs]));
            end

            this.numWeights = numHidden * (numInputs + numOutputs + 1) + numOutputs;   % MLP
            this.gbest = zeros(this.numWeights, 2);
            
            pop = (rand(this.numWeights, 2, popsize) - 0.5) * 2;   % [-1,1]

            this.popsize        = popsize;
            this.pop            = pop;
            this.numInputs      = numInputs;
            this.numOutputs     = numOutputs;
            this.numHidden      = numHidden;
        end
        
        %% 
        % DE-PSO  training
        %
        function gbeste = train(this, data)            
            if nargin < 2 || (size(data, 2) ~= this.numInputs + this.numOutputs)
               error('No data or wrong number of columns.');
            end 
                     
            this.initPlot();
            
            this.dataset = data;
            this.numTrainingSamples = size(data, 1);
             
            % TODO try to replace dimensions with 1
            this.pop_hist = zeros(this.numWeights, 2, this.popsize, 2);
            this.err_hist = zeros(this.popsize, 2);
            this.gbest_hist = zeros(this.numWeights, 2);
            this.gbeste_hist = zeros(2, 1);
                
                                
            % check initial pop                                                    
            this.iGen = 1;
            for i = 1 : this.popsize                
                x = this.pop(:, :, i);
                E = this.err(x);                    
                if E < this.gbeste
                   this.gbeste = E;
                   this.gbest = x;
                end
                this.err_hist(i, this.iGen) = E;
            end
            this.pop_hist(:, :, :, this.iGen) = this.pop;
            this.gbest_hist(:, this.iGen) = this.gbest(:, 1);
            this.gbeste_hist(this.iGen) = this.gbeste;
            
            fprintf('\nGeneration %d: Best Error = %0.10f\n', this.iGen, this.gbeste);                                        
            fprintf('Mean: %0.10f\n', mean(this.err_hist(:, this.iGen)));
            fprintf('Std.: %0.10f\n', std(this.err_hist(:, this.iGen)));
            
            this.updatePlot();
                
            %this.dist = (1 - 1 / (1 + (this.gbeste / this.termErr))) * (1 - abs(log10(this.termErr))/15);
            
            % other generations
            while ~this.isComplete()
                this.iGen = this.iGen + 1;
                
                % Reduce entropy
                this.S = this.S * (1 - this.alpha);
                                
                for i = 1 : this.popsize   
                    this.iParticle = i;
                    
                    x = this.pop(:, :, i);                    
                    x = this.depso_x(x);                     
                    this.pop(:, :, i) = x;

                    if this.canUseEx
                        E = this.Ex;
                    else
                        E = this.err(x);
                    end               
                   
                    if E < this.gbeste
                       this.gbeste = E;
                       this.gbest = x;
                       %this.dist = (1 - 1 / (1 + (this.gbeste / this.termErr))) * (1 - abs(log10(this.termErr))/15);
                       %disp(this.dist);
                    end                   
                    this.err_hist(i, this.iGen) = E;
                end
                this.pop_hist(:, :, :, this.iGen) = this.pop;
                this.gbest_hist(:, this.iGen) = this.gbest(:, 1);
                this.gbeste_hist(this.iGen) = this.gbeste;
                
                this.imp = circshift(this.imp, [1 1]);
                this.imp(1) = (this.gbeste_hist(this.iGen - 1) - this.gbeste) / this.gbeste_hist(this.iGen - 1);
                                
                fprintf('\nGeneration %d: Best Error = %0.10f\n', this.iGen, this.gbeste);                                        
                fprintf('Mean: %0.10f\n', mean(this.err_hist(:, this.iGen)));
                fprintf('Std.: %0.10f\n', std(this.err_hist(:, this.iGen)));  
                
                this.updatePlot();
            end
            
            this.gbest = this.local_search(this.gbest, 1);                            
            gbeste = this.Ex;
            fprintf('\nBest Error: %0.10f\n', gbeste);
        end   
        
        %%
        % Stop condition
        %
        function result = isComplete(this)
           result = 0;          
           if (this.iGen >= this.termGen)
               fprintf('\nGeneration limit reached.\n');
               result = 1;
           end
           if (this.gbeste < this.termErr) 
               fprintf('\nError threshold reached.\n');
               result = 1;
           end
           if this.S < this.termImpS && sum(this.imp) < this.termImp
               fprintf('\nImprovement threshold reached.\n');
               result = 1;
           end
           if std(this.err_hist(:, this.iGen)) < this.termConv
               fprintf('\nConvergence threshold reached.\n');
               result = 1;
           end
        end                

        %%
        % DE-PSO for one particle
        %
        function x = depso_x(this, x)            
            L = this.Lmin + (this.Lmax - this.Lmin) * (1 - this.S);            
            if rand < L
                x = this.local_search(x);
                this.canUseEx = 1;
            elseif rand < .5
                x = this.pso(x);
                this.canUseEx = 0;
            else                           
                x = this.de(x);
                this.canUseEx = 1;
            end
        end
        
        %%
        % DE for one particle
        %
        function x = de(this, x)
            z = this.de_mutate(x);
            u = this.de_cross(x, z);
            this.Ex = this.err(x);
            Eu = this.err(u);
            if Eu < this.Ex
                x = u;
                this.Ex = Eu;
            end
        end
        
        %%
        % DE Mutate
        %
        function z = de_mutate(this, x)            
            F = this.Fmin + (this.Fmax - this.Fmin) * this.S;
            xr = this.randX(this.pop, 4, this.iParticle);
            %z = x;
            z(:, 1) = this.gbest(:, 1) + F * (xr(:, 1, 1) - xr(:, 1, 2) + xr(:, 1, 3) - xr(:, 1, 4));
            z(:, 2) = x(:, 2) * 0;
            %z = this.gbest + F * (xr(:, :, 1) - xr(:, :, 2) + xr(:, :, 3) - xr(:, :, 4));
        end

        %%
        % DE Crossover
        %
        function u = de_cross(this, x, z)            
            CR = this.CRmin + (this.CRmax - this.CRmin) * this.S;            
            u = x;            
            wr = randperm(this.numWeights, 1);    
            for i = 1 : this.numWeights
                if rand < CR || wr == i
                    u(i, 1) = z(i, 1);               
                end
            end            
        end
        
        %%
        % Random position vectors
        %
        function xr = randX(this, pop, n, iAvoid)  % current i to avoid
            if nargin < 3
                n = 1;
            end
            if nargin < 4
                iAvoid = 0;
            end
            
            r = randperm(this.popsize, n + 1);  % get an extra index in case iAvoid is selected
            xr = zeros(this.numWeights, 2, n);
                        
            i = 1;
            count = 1;
            while count <= n
                if r(i) ~= iAvoid
                    xr(:, :, count) = pop(:, :, r(i));                    
                    count = count + 1;
                end
                i = i + 1;
            end
        end

        %%
        % Difference of two random particles
        %
        function d = diff(this, pop)
            xr = this.randX(pop, 2);            
            d = xr(:, :, 1) - xr(:, :, 2);
        end       
                
        %%
        % PSO for one particle
        %
        function x = pso(this, x)                        
            pbest = this.diff(this.pop); % DE operation              
            %x(:, 2) = this.w * x(:, 2) +  this.c1 * rand * (pbest(:, 1) - x(:, 1)) + this.c2 * rand * (this.gbest(:, 1) - x(:, 1));             
            x(:, 2) = this.w * x(:, 2) * this.S + this.c1 * pbest(:, 1) * this.S + this.c2 * (rand/2 + .5) * (this.gbest(:, 1) - x(:, 1));             
            vmax = this.VmaxMin + (this.VmaxMax - this.VmaxMin) * this.S;
            for i = 1 : this.numWeights
                if x(i, 2) > vmax;
                    x(i, 2) = vmax;
                elseif x(i, 2) < (0 - vmax)
                    x(i, 2) = (0 - vmax);
                end
            end
            x(:, 1) = x(:, 1) + x(:, 2);                        
        end
        
        %% 
        % Error of one particle (all samples)
        %
        function E = err(this, x)
            if size(x, 1) == 1
                x = x'; % using gatool
            end
            E = 0;
            numSamples = size(this.dataset, 1);
            for i = 1 : numSamples
                Xdata = this.dataset(i, 1:this.numInputs);
                yStart = this.numInputs + 1;
                yEnd   = yStart + this.numOutputs - 1;
                Ydata = this.dataset(i, yStart:yEnd);

                Ynn = this.calcYnn(Xdata, x);
                            
                for k = 1 : this.numOutputs
                   E = E + .5 * (Ynn(k) - Ydata(k)) ^ 2;
                end            
            end
            E = E / numSamples;
        end
        
        %%
        % Calculate Ynn for one sample
        %
        function [Ynn z] = calcYnn(this, Xdata, x)              
            aEnd = this.numHidden * this.numInputs;
            bEnd = aEnd + this.numHidden;
            cEnd = bEnd + this.numHidden * this.numOutputs;
            
            Ynn = zeros(this.numOutputs);                        
            gamma = zeros(this.numHidden);
            z     = zeros(this.numHidden);              

            for i = 1 : this.numInputs
                for j = 1 : this.numHidden                
                    gamma(j) = gamma(j) + x((i - 1) * j + j, 1) * Xdata(i);
                end
            end

            for j = 1 : this.numHidden
                gamma(j) = gamma(j) + x(aEnd + j, 1);      % biases
                z(j)     = 1 / (1 + exp(-gamma(j)));

                for k = 1 : this.numOutputs                   
                    Ynn(k) = Ynn(k) + x(bEnd + (j - 1) * k + k, 1);
                end
            end

            for k = 1 : this.numOutputs
               Ynn(k) = Ynn(k) + x(cEnd + k, 1);           % biases               
            end                            
        end
        

        %%
        % use BP or conjugate gradient to find the lowest (acceptable) point
        %
        function x = local_search(this, x, extended)
            v = x(:, 2);  
            x = x(:, 1);               
            
            limit = this.lsEpochsExtended;
            if nargin < 3  
                limit = this.lsEpochs;
            end
              
            for i = 1 : limit                                                 
                x_1 = x;
                derivs = this.calcDerivs(x);                
                a = this.calcAlpha(x, derivs);
                x = x - derivs * a;              
                derivs_1 = derivs;
                derivs = this.calcDerivs(x);
                B = (derivs' * derivs) / (derivs_1' * derivs_1);              
                derivs = -1 * derivs + B * -1 * derivs_1;
                x = x + a * derivs + a * B * (x - x_1);                
                this.Ex = this.err(x);                
                if (this.Ex < this.termErr || mean(derivs) < this.lsDerivThreshold)
                    break;
                end
            end
            
            x(:, 2) = v;
        end
        
        function derivs = calcDerivs(this, x)            
            derivs = zeros(size(x));
            
            aEnd = this.numHidden * this.numInputs;
            bEnd = aEnd + this.numHidden;
            cEnd = bEnd + this.numHidden * this.numOutputs;
            
            numSamples = size(this.dataset, 1);
            for s = 1 : numSamples
                Xdata = this.dataset(s, 1:this.numInputs);
                yStart = this.numInputs + 1;
                yEnd   = yStart + this.numOutputs - 1;
                Ydata = this.dataset(s, yStart:yEnd);

                [Ynn z] = this.calcYnn(Xdata, x);
                            
                for k = 1 : this.numOutputs
                   
                   derivs(cEnd + k) = derivs(cEnd + k) + (Ynn(k) - Ydata(k));
                   
                    for j = 1 : this.numHidden
                        derivs(bEnd + (j - 1) * k + k) = derivs(bEnd + (j - 1) * k + k) + derivs(cEnd + k) * z(j);
                    
                        derivs(aEnd + j) = derivs(aEnd + j) + derivs(cEnd + k) * derivs(bEnd + (j - 1) * k + k) * z(j) * (1 - z(j));
                        
                        for i = 1 : this.numInputs
                            derivs((i - 1) * j + j) = derivs((i - 1) * j + j) + ...
                                derivs(cEnd + k) * derivs(bEnd + (j - 1) * k + k) * z(j) * (1 - z(j)) * Xdata(i);
                        end
                    end
                end 
            end
            derivs = derivs / numSamples;
        end
        
        function alpha = calcAlpha(this, x, derivs)            
            % Using linear search algorithm to find where alpha converges.
            % The magnitude of alpha values are always in the order
            % a1 ... a3 ... a4 ... a2
            a1 = 0;
            a2 = 1;                     
            
            a3 = a2 - 0.618 * (a2 - a1);
            a4 = a1 + 0.618 * (a2 - a1);

            E3 = this.err(x - derivs * a3);
            E4 = this.err(x - derivs * a4);
            
            epsilon = .05;
            
            while 1
                if E3 > E4
                    a1 = a3;
                    a3 = a4;
                    a4 = a1 + 0.618 * (a2 - a1);
                    
                    if (a2 - a1) <= epsilon
                        alpha = a3; % or any a#  
                        break;
                    end
                    
                    E3 = E4;
                    E4 = this.err(x - derivs * a4);
                else
                    a2 = a4;
                    a4 = a3;
                    a3 = a2 - 0.618 * (a2 - a1);
                    
                    if (a2 - a1) <= epsilon
                        alpha = a3; % or any a#  
                        break;
                    end
                    
                    E4 = E3;
                    E3 = this.err(x - derivs * a3);
                end
            end
        end
        
        %%
        % Initialize plot
        %
        function initPlot(this)
            close;
            axis([0, Inf, 0, Inf]);
            xlabel('Generation');
            ylabel('Fitness');            
        end
               
        function updatePlot(this)
           hold on                
            plot(this.iGen, mean(this.err_hist(:, this.iGen)), 'bd', 'MarkerFaceColor', 'b', 'MarkerSize', 5);                
            plot(this.iGen, this.gbeste, 'kd', 'MarkerFaceColor', 'k', 'MarkerSize', 5); 
            legend('Mean', 'Best');
            title(['Mean: ', num2str(mean(this.err_hist(:, this.iGen))), '    Best: ', num2str(this.gbeste)]);                    
            drawnow;
            hold off 
        end
    end
end