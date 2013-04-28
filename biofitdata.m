function data = biofitdata(~)
    x = -2:.01:2;    
    len = length(x);
    y = zeros(len,1);
    for i = 1 : len
        y(i) = (-2^(-2*((x(i)-0.1)/0.9)^2)*(sin(5*pi*x(i))^6)) + 1;
    end
    data = zeros(len,2);
    data(:, 1) = x';
    data(:, 2) = y';
end
