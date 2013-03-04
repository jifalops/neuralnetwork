function weights = calcnn(data, weights)
   data = [1 2 4; 1 3 5];
   % u11 u12 u21 u22 u01 u02 v11 v21 v01
   weights = [1 1 1 1 1 1 1 1 1];      
   
   epochs = 100;
   
   yerror = zeros(2, epochs);
   
   for i = 1 : epochs
       fprintf('\nEpoch %d:\n', i);
       for j = 1 : 2
        ynn = calc_ynn(data, weights, j);       
        yerror(j, i) = calcYerror(ynn, data(j, 3));
        d = calcDerivs(data, weights, j);
        weights = update_weights(weights, d);
        fprintf('\nSample %d:\n', j);
        fprintf('Error: %f\n', yerror(j, i));
        fprintf('Weights:\n'), fprintf('%f\n', weights);
        %fprintf('Derivs :'), fprintf('%f', d), fprintf('\n');
      end
   end
   subplot(2, 1, 1), plot(yerror(1, :))
   subplot(2, 1, 2), plot(yerror(2, :))
end

function ynn = calc_ynn(data, w, sample_num)
    %calc ynn
    g1 = w(5) + w(1)*data(sample_num,1) + w(3)*data(sample_num,2);
    g2 = w(6) + w(2)*data(sample_num,1) + w(4)*data(sample_num,2);
    z1 = 1/(1+exp(-g1));
    z2 = 1/(1+exp(-g2));
    ynn = w(9) + w(7)*z1 + w(8)*z2;
end

function yerror = calcYerror(ynn, ytable)
  yerror = .5*(ynn-ytable)^2;
end

function d = calcDerivs(data, w, sample_num)
    %calc ynn
    g1 = w(5) + w(1)*data(sample_num,1) + w(3)*data(sample_num,2);
    g2 = w(6) + w(2)*data(sample_num,1) + w(4)*data(sample_num,2);
    z1 = 1/(1+exp(-g1));
    z2 = 1/(1+exp(-g2));
    ynn = w(9) + w(7)*z1 + w(8)*z2;
    
    dy = ynn - data(sample_num, 3);

    % u11
    d(1) = dy * w(7) * z1 * (1-z1) * data(sample_num,1);
    
    % u12
    d(2) = dy * w(8) * z2 * (1-z2) * data(sample_num,1);
    
    % u21
    d(3) = dy * w(7) * z1 * (1-z1) * data(sample_num,2);
    
    % u22
    d(4) = dy * w(8) * z2 * (1-z2) * data(sample_num,2);
    
    % u01
    d(5) = dy * w(7) * z1 * (1-z1);
    
    % u02
    d(6) = dy * w(8) * z2 * (1-z2);
    
    % v11
    d(7) = dy * z1;
    
    % v21
    d(8) = dy * z2;
    
    % v01
    d(9) = dy;
    
end

function weights = update_weights(weights, derivatives)
    for i = 1 : 9
        weights(i) = weights(i) + (-derivatives(i)*.1);
    end
end