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
% dr        DE crossover random dimension index
% pbest     Particle's best known position
% gbest     Swarm's best known position
% w         PSO inertia weight
% c1        PSO cognitive factor
% c2        PSO social factor
% S         Entropy
% alpha     Entropy velocity [0,1]
%
% pop_hist      Population history
% err_hist      Error/fitness history
% gbest_hist    Global best history

% Input/Output Data Matrix
%
%   x-dimension: Each row is a sample
%   y-dimension: Each column is an input or output variable
%   z-dimension: Dimension index (objective sampled)

% Population Matrix (with history)
%
%   x-dimension: Each row consists of two weights
%   y-dimension: Column 1 = position weights, 
%                Column 2 = velocity weights
%   z-dimension: Dimension index (objective being measured)
% 4th-dimension: Each is a particle/individual (has position and velocity in d dimensions)
% 5th-dimension: Generation/epoch index


% Error/Fitness Matrix (with history)
%
%   x-dimension: Each row consists of two numbers
%   y-dimension: Column 1 = particle index (from population), 
%                Column 2 = error/fitness value
%   z-dimension: Dimension index
% 4th-dimension: Generation/epoch index


% Global best matrix (with history)
%
%   x-dimension: Each row represents gbest for a particular dimension/objective
%   y-dimension: Column 1 = particle index (from population), 
%                Column 2 = error/fitness value
%   z-dimension: Generation/epoch index

%%
classdef Depso < handle
    properties (Constant)
        % PSO weights
        w  = 0.7298
        c1 = 1.496
        c2 = 1.496
    end    
    properties % read and write access
       alpha = 0.1;
       
       Fmax  = 2;
       Fmin  = 0.2;   
       Lmax = .1;
       Lmin = .01;
       CRmin = .1;
       CRmax = 1;
       Vmax  = 2;
              
       batchMode = 0;
       popsize = 10;                        
       numObjectives    = 1;   
       lsMaxEpochs      = 3;
       numTrainingSamples;
       numTestSamples;
       numInputs;
       numOutputs;
       numHidden;    
       pop;
       curGen;   
       
       % Termination conditions (any will stop)
       termGen    = 50;
       termErr    = 0.001;
       termImp    = 0.05;   % Avg. improvement
       termImpGen = 5;      % over x generations 
    end    
    properties (SetAccess = private) % read-only for user
       S          = 1;       
       pop_hist   = zeros(1,1,1,1,1);
       err_hist   = zeros(1,1,1,1);
       gbest_hist = zeros(1,1,1);
       numWeights;
    end
    
    properties (Hidden)        
        curSample;
        curParticle;     
        curDim;
        data;        
    end
 
    %%
    methods
        %%
        % Initialize population
        %
        function pop = initPop(this, popsize, numInputs, numOutputs, numHidden, numObjectives) 
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
            if nargin < 6 || numObjectives < 1
               numObjectives = this.numObjectives; 
            end
            
            % MLP
            numWeights = numHidden * (numInputs + numOutputs + 1) + numOutputs;
            
            pop = (rand(numWeights, 2, numObjectives, popsize) - 0.5) * 2;   % [-1,1]

            this.numWeights     = numWeights;
            this.popsize        = popsize;
            this.pop            = pop;
            this.numInputs      = numInputs;
            this.numOutputs     = numOutputs;
            this.numHidden      = numHidden;   
            this.numObjectives  = numObjectives;
        end
        
        %% 
        % DE-PSO  training
        %
        function gbest = train(this, data)            
            if nargin < 2 || (size(data, 2) ~= this.numInputs + this.numOutputs)
               error('No data or wrong number of columns.');
            end 
            
            this.numTrainingSamples = size(data, 1);  % same number of samples in each objective
            this.data = data;
            
            if this.numObjectives == 1
               data(:, :, 1) = data;
            end
            
            popsize       = this.popsize;
            pop           = this.pop;
            numInputs     = this.numInputs;
            numOutputs    = this.numOutputs;
            numHidden     = this.numHidden;
            numObjectives = this.numObjectives;
            
            S             = this.S;
            
            gbest = zeros(this.numObjectives, 2);
            for d = 1 : numObjectives
                r = randperm(popsize, 1);                
                this.curParticle = r;    
                this.curSample = randperm(this.numTrainingSamples, 1);
                gbest(d, 1) = r;
                gbest(d, 2) = this.err(pop(:, 1, d, r));   % random particle for each dimension
            end

            
            if this.batchMode
                
            else
                for k = 1 : this.termGen        % Generation limit  
                    this.curGen = k;
                    for sample = 1 : this.numTrainingSamples
                        this.curSample = sample;
                        for d = 1 : numObjectives
                            this.curDim = d;
                            for i = 1 : popsize     % For each particle 
                               this.curParticle = i;                               
                               x = pop(:, :, d, i);
                               x = this.depso_x(pop, x, S, gbest);                  
                            end
                        end
                    end

                    % temporary measure to keep entropy >= 0. entropy * alpha is also an option.
                    S = abs(S - alpha); 
                end
            end
            
        end                

        %%
        % DE-PSO for one particle
        %
        function x = depso_x(this, pop, x, S, gbest)            
            L = this.Lmin + (this.Lmax - this.Lmin) * (1 - S);            
            if rand < L
                x = this.local_search(x);
            elseif rand < .5
                x = this.pso(pop, x, gbest);
            else                
                x = this.de(pop, x, S);
            end
        end
        
        %%
        % DE for one particle
        %
        function x = de(this, pop, x, S)
            z = this.de_mutate(pop, x, S);
            u = this.de_cross(x, z, S);
            Ex = this.err(x(:, 1, :));
            Eu = this.err(u(:, 1, :));
            if Eu < Ex
                x(:, 1, :) = u;    
            end
        end
        
        %%
        % DE Mutate
        %
        function z = de_mutate(this, pop, x, S)            
            F = this.Fmin + (this.Fmax - this.Fmin) * S;
            xr = this.randX(pop, 3, this.curParticle);
            z = x;
            z(:, 1, :) = xr(:, 1, :, 1) + F * (xr(:, 1, :, 2) - xr(:, 1, :, 3));
        end

        %%
        % DE Crossover
        %
        function u = de_cross(this, x, z, S)            
            CR = this.CRmin + (this.CRmax - this.CRmin) * S;            
            u = zeros(this.numWeights, 1, this.numObjectives);            
            for d = 1 : this.numObjectives
                dr = randperm(this.numWeights, 1);    % Using random weight instead of dimension
                for i = 1 : this.numWeights
                    if rand < CR || dr == i
                        u(i, 1, d) = z(i, 1, d);
                    else
                        u(i, 1, d) = x(i, 1, d);
                    end
                end
            end
        end
        
        %%
        % Random position vectors
        %
        function xr = randX(this, pop, n, ci)  % current i
            if nargin < 3
                n = 1;
            end
            if nargin < 4
                ci = 0;
            end                        
            r = randperm(this.popsize, n + 1);
            xr = zeros(this.numWeights, 1, this.numObjectives, n);
            
            i = 1;
            count = 1;
            while count <= n
                if r(i) ~= ci
                    xr(:, 1, :, count) = pop(:, 1, :, r(i));
                    count = count + 1;
                end
                i = i + 1;
            end
        end

        %%
        % Difference of two random positions
        %
        function d = diff(this, pop)
            xr = this.randX(pop, 2);            
            d = xr(:, 1, :, 1) - xr(:, 1, :, 2);
        end       

        %%
        % PSO for one particle
        %
        function x = pso(this, pop, x, gbest)            
            for d = 1 : this.numObjectives                
                pbest = diff(pop); % DE operation                 
                x(:, 2) = x(:, 2) * 0.7298  +  pbest(:, 1) * 1.496 + pop(:, 1, d, gbest(d, 1)) * 1.496; 
                x(:, 1, d) = x(:, 1, d) + x(:, 2, d);	
                Ex = this.err(x);                
                if (Ex < gbest(d, 2))
                    gbest(d, 1) = this.curParticle;
                    gbest(d, 2) = Ex;
                end
            end
        end
        
        %% 
        % Error of one sample
        %
        function E = err(this, w)
            Xdata = this.data(this.curSample, 1:this.numInputs, :);
            yStart = this.numInputs + 1;
            yEnd   = yStart + this.numOutputs - 1;
            Ydata = this.data(this.curSample, yStart:yEnd, :);
            
            Ynn = this.calcYnn(Xdata, w);
            
            
            E = zeros(this.numObjectives, 1);
            for d = 1 : this.numObjectives
                for k = 1 : this.numOutputs
                   E(d) = E(d) + .5 * (Ynn(1, k, d) - Ydata(1, k, d)) ^ 2;
                end
            end
        end
        
        %%
        % Calculate Ynn for one sample, multiple dimensions/objectives
        %
        function Ynn = calcYnn(this, Xdata, w)  
            aEnd = this.numHidden * this.numInputs;
            bEnd = aEnd + this.numHidden;
            cEnd = bEnd + this.numHidden * this.numOutputs;
            
            Ynn = zeros(1, this.numOutputs, this.numObjectives);
            
            for d = 1 : this.numObjectives
                gamma = zeros(this.numHidden);
                z     = zeros(this.numHidden);  
                y   = zeros(this.numOutputs);

                for i = 1 : this.numInputs
                    for j = 1 : this.numHidden                
                        gamma(j) = gamma(j) + w((i - 1) * j + j, 1, d) * Xdata(1, i, d);
                    end
                end

                for j = 1 : this.numHidden
                    gamma(j) = gamma(j) + w(aEnd + j, 1, d);      % biases
                    z(j)     = 1 / (1 + exp(-gamma(j)));

                    for k = 1 : this.numOutputs                   
                        y(k) = y(k) + w(bEnd + (j - 1) * k + k, 1, d);
                    end
                end

                for k = 1 : this.numOutputs
                   y(k) = y(k) + w(cEnd + k, 1, d);           % biases
                   Ynn(1, k, d) = y(k);
                end                
            end
        end

        %%
        % use BP or conjugate gradient to find the lowest (acceptable) point
        %
        function x = local_search(this, x)

        end
    end
end