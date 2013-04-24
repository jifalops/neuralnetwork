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
% alpha     Entropy velocity [0,1]
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
%   x-dimension: Each row consists of two numbers
%   y-dimension: Column 1 = particle index (from population), 
%                Column 2 = error/fitness value
%   z-dimension: Generation/epoch index


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
        % PSO weights
        w  = 0.7298
        c1 = 1.496
        c2 = 1.496
    end    
    properties % read and write access
       Fmax  = 1.5;
       Fmin  = 0.2;   
       Lmax = .1;
       Lmin = .01;
       CRmin = .1;
       CRmax = 1;
       Vmax  = 2;
              
       batchMode = 0;
       popsize = 20;          
       lsMaxEpochs = 3;
       numTrainingSamples;
       numTestSamples;
       numInputs;
       numOutputs;
       numHidden;    
       pop;
       gbest;
       gbeste = 999999999;
       iGen = 1;   
       
       alpha = 0.05;
       
       % Termination conditions (any will stop)
       termGen    = 50;
       termErr    = 0.01;
       termImp    = 0.01;   % Avg. improvement
       termImpGen = 5;      % over x generations 
       imp = ones(1, 5); % same num
    end    
    properties (SetAccess = private) % read-only for user
       S          = 1;       
       pop_hist   = zeros(1,1,1,1,1);
       err_hist   = zeros(1,1,1,1);
       gbest_hist = zeros(1,1,1);
       numWeights;
    end
    
    properties (Hidden)        
        iSample = 1;
        iParticle = 1;
        dataset;  
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
            
            this.dataset = data;
            this.numTrainingSamples = size(data, 1);

                %r = randperm(this.popsize, 1);                
                %this.iParticle = r;    
                %this.iSample = randperm(this.numTrainingSamples, 1);
                %gbest(d, 1) = r;
                %gbest(d, 2) = this.err(pop(:, 1, d, r));   % random particle for each dimension
             
            
            if this.batchMode
                error('no soup for you');
            else
                while ~this.isComplete()                                    
                    gbeste_tmp = this.gbeste;
                    for s = 1 : this.numTrainingSamples
                        this.iSample = s;                       
                        for i = 1 : this.popsize     % For each particle 
                            this.iParticle = i;
                            x = this.pop(:, :, i);
                            x = this.depso_x(x); 
                                                        
                            if this.canUseEx
                                E = this.Ex;
                            else
                                E = this.err(x);
                            end               
                                                        
                            if E < this.gbeste
                               this.gbest = x;
                               this.gbeste = E;
                            end
                            
                            this.pop(:, :, i) = x;
                        end                         
                    end
                    
                    this.imp = circshift(this.imp, [1 1]);
                    this.imp(1) = abs(gbeste_tmp - this.gbeste) / gbeste_tmp;
                    
                    fprintf('Generation %d: E = %f\n', this.iGen, this.gbeste);                                        

                    % temporary measure to keep entropy >= 0. entropy * alpha is also an option.
                    this.S = abs(this.S - this.alpha); 
                    this.iGen = this.iGen + 1;
                end
            end
            gbeste = this.gbeste;
        end   
        
        function result = isComplete(this)
           result = 0;          %disp(this.iGen);disp(this.termGen);disp(this.gbeste);disp(this.termErr);disp(mean(this.imp));disp(this.termImp);
           if (this.iGen > this.termGen) || (this.gbeste < this.termErr) ...
                   || ((mean(this.imp) < this.termImp) && this.S < .3)
               result = 1;
           end
        end

        %%
        % DE-PSO for one particle
        %
        function x = depso_x(this, x)            
            L = this.Lmin + (this.Lmax - this.Lmin) * (1 - this.S);            
           % if rand < L
            %    x = this.local_search(x);
            %elseif
            if rand < .5
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
            xr = this.randX(this.pop, 3, this.iParticle);
            z = x;
            z(:, 1) = xr(:, 1, 1) + F * (xr(:, 1, 2) - xr(:, 1, 3));
        end

        %%
        % DE Crossover
        %
        function u = de_cross(this, x, z)            
            CR = this.CRmin + (this.CRmax - this.CRmin) * this.S;            
            u = x;            
            wr = randperm(this.numWeights, 1);    % Using random weight instead of dimension
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
            x(:, 2) = x(:, 2) * 0.7298  +  pbest(:, 1) * 1.496 + this.gbest(:, 1) * 1.496; 
            x(:, 1) = x(:, 1) + x(:, 2);            
        end
        
        %% 
        % Error of one sample
        %
        function E = err(this, w)
            Xdata = this.dataset(this.iSample, 1:this.numInputs);
            yStart = this.numInputs + 1;
            yEnd   = yStart + this.numOutputs - 1;
            Ydata = this.dataset(this.iSample, yStart:yEnd);
            
            Ynn = this.calcYnn(Xdata, w);
                        
            E = 0;            
            for k = 1 : this.numOutputs
               E = E + .5 * (Ynn(k) - Ydata(k)) ^ 2;
            end            
        end
        
        %%
        % Calculate Ynn for one sample
        %
        function Ynn = calcYnn(this, Xdata, x)  
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
        function x = local_search(this, x)

        end
    end
end