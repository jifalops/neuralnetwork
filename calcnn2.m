function weights = calcnn2(data, weights)
   data = [1 2 4; 1 3 5];
   % u11 u12 u21 u22 u01 u02 v11 v21 v01
   weights = [1 1 1 1 1 1 1 1 1];      
   
   epochs = 200; 
   
   yerror = zeros(1, epochs);
   
   for i = 1 : epochs
       fprintf('\nEpoch %d:\n', i);
        ynn = calc_ynn(data, weights);       
        yerror(i) = calcYerror(ynn, data(:, 3));
        d = calcDerivs(data, weights);
        weights = update_weights(weights, d);
        fprintf('Error: %f\n', yerror(i));
        fprintf('Weights:\n'), fprintf('%f\n', weights);    
        %fprintf('Derivs :'), fprintf('%f', d), fprintf('\n');
   end
  plot(yerror)

end

function ynn = calc_ynn(data, w)
    ynn = zeros(1,2);
    for i = 1 : 2
        g1 = w(5) + w(1)*data(i,1) + w(3)*data(i,2);
        g2 = w(6) + w(2)*data(i,1) + w(4)*data(i,2);
        z1 = 1/(1+exp(-g1));
        z2 = 1/(1+exp(-g2));
        ynn(i) = w(9) + w(7)*z1 + w(8)*z2;
    end
end

function yerror = calcYerror(ynn, ytable)
    yerror = 0;
    for i = 1 : 2
        yerror = yerror + .5*(ynn(i)-ytable(i))^2;
    end
end

function d = calcDerivs(data, w)
    d = zeros(1, 9);
    for i = 1 : 2
        %calc ynn
        g1 = w(5) + w(1)*data(i,1) + w(3)*data(i,2);
        g2 = w(6) + w(2)*data(i,1) + w(4)*data(i,2);
        z1 = 1/(1+exp(-g1));
        z2 = 1/(1+exp(-g2));
        ynn = w(9) + w(7)*z1 + w(8)*z2;

        dy = ynn - data(i, 3);

        % u11
        d(1) = d(1) + dy * w(7) * z1 * (1-z1) * data(i,1);

        % u12
        d(2) = d(2) + dy * w(8) * z2 * (1-z2) * data(i,1);

        % u21
        d(3) = d(3) + dy * w(7) * z1 * (1-z1) * data(i,2);

        % u22
        d(4) = d(4) + dy * w(8) * z2 * (1-z2) * data(i,2);

        % u01
        d(5) = d(5) + dy * w(7) * z1 * (1-z1);

        % u02
        d(6) = d(6) + dy * w(8) * z2 * (1-z2);

        % v11
        d(7) = d(7) + dy * z1;

        % v21
        d(8) = d(8) + dy * z2;

        % v01
        d(9) = d(9) + dy;
    end    
end

function weights = update_weights(weights, derivatives)
    for i = 1 : 9
        weights(i) = weights(i) + (-derivatives(i)*.1);
    end
end