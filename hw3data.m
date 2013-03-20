%% To visualize use: plot(data(:, 1), data(:, 2))
function data = hw3data() 
    min = 0;
    max = 4;    
    scale = 200;
    rows = (max - min) * scale;
    
    data = zeros(rows, 2);
    for i = 1 : rows
       x = i / scale;
       
       if x >= 0.9 && x <= 1.1
           y = 50 + 50 * sin((x-0.9)*2*pi*10);
       elseif x >= 2.9 && x <= 3.1
           y = 50 + 50 * sin((x-2.9)*2*pi*10);
       else
           y = 50;
       end
       
       data(i, 1) = x;
       data(i, 2) = y;
    end
end