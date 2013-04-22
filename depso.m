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
% alpha     Entropy velocity
%
% pop_hist      Population history
% err_hist      Error/fitness history
% gbest_hist    Global best history


% Population Matrix (with history)
%
%   x-dimension: Each row consists of weights
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
classdef depso < handle
    properties (Constant)
        % PSO weights
        w  = 0.7298
        c1 = 1.496
        c2 = 1.496
    end    
    properties % user can read and write values
       Fmax = 2;
       Fmin = 0;      
       Vmax = 2;
    end    
    properties (SetAccess = private)
       S          = 1;
       pop_hist   = zeros(1,1,1,1,1);
       err_hist   = zeros(1,1,1,1);
       gbest_hist = zeros(1,1,1);
    end
    
    %%
    methods (Static)
        function pop = initPop(popsize, d) % d = # of dimensions
            
        end
    end

    %%
    methods
        %% run DE-PSO 
        function gbest = depso(this, popsize, dim, alpha) % alpha = entropy change rate [0,1]
            if nargin < 2
               popsize = 20;
            end
            if nargin < 3
               dim = 1; 
            end
            if nargin < 4
               alpha = 0.1; 
            end
                        
            S     = 1;
            gbest = -1;    

            
            
            pop_size = size(pop, 3);
            dimensions = size(pop, 4);

            % 100 generations
            for k = 1 : 100
                for d = 1 : dimensions
                    for i = 1 : pop_size       

                       pop(:, :, i, d) = depso_x(pop, pop(:, :, i, d), gbest, entropy);                   
                    end
                end
                entropy = abs(entropy - alpha); % temporary measure to keep entropy >= 0. entropy * alpha is also an option.
            end
        end                

        % handle one particle
        function x = depso_x(pop, x, gbest, entropy)
            Lmax = .1;
            Lmin = .01;
            L = Lmin + (Lmax - Lmin) * (1 - entropy);

            r = rand;
            if r < L
                x = local_search(x);
            elseif r < .5
                x = pso(pop, x, gbest);
            else
                x = de(pop, x, entropy);
            end
        end

        % Error of one weight vector (position)
        function E = err(x)

        end

        % use BP or conjugate gradient to find the lowest (acceptable) point
        function x = local_search(x)

        end

        % DE for one particle
        function x = de(pop, x, entropy)
            z = de_mutate(pop, x, entropy);
            u = de_cross(x, z, entropy);
            Ex = err(x);
            Eu = err(u);
            if Eu < Ex
                x = u;    
            end
        end

        function x = pso(pop, x, gbest) 
            gbest_temp = gbest;
            if gbest == inf 
                gbest_temp = 0;
            end
            d = diff(pop);    
            v = v * 0.7298  +  d * 1.496 + gbest_temp * 1.496; % d = pbest_perturbed
            x = x + v;	
            Ex = err(x);
            Egbest = err(gbest_tmp);	% TODO gbest error should be remembered with gbest.
            if (Ex < Egbest)
                gbest = x;
            end
        end

        function xr = randPos(pop, n)
            if nargin < 2
                n = 1;
            end
            len = size(pop, 3);
            r = randperm(len, n);
            xr = zeros(1, 1, n);    % x random
            for i = 1 : n
               xr(:, :, i) = pop(:, :, r(i)); 
            end
        end

        function d = diff(pop)
            xr = randPos(pop, 2);
            d = xr(:, 1, 1) - xr(:, 1, 2);
        end

        function z = de_mutate(pop, x, entropy)
            Fmax = 2;
            Fmin = 0.2;
            F = Fmin + (Fmax - Fmin) * entropy;

            xr = randPos(pop, 3);
            z = x;
            z(:, 1) = xr(:, 1, 1) + F * (xr(:, 1, 2) - xr(:, 1, 3));
        end

        function u = de_cross(x, z, entropy)
            CRmin = .1;
            CRmax = 1;
            CR = CRmin + (CRmax - CRmin) * entropy;

            len = size(x, 1);
            u = zeros(len);
            for i = 1 : len
                if (rand < CR) % or d == rni
                    u(i, 1) = z(i, 1);
                else
                    u(i, 1) = x(i, 1);
                end
            end
        end

    end
end