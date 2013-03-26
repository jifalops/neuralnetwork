function data = varAlphaData(~)
    x = -2:.1:2;
    for i = 1 : length(x)
        y1(i) = -x(i) + x(i)^2 - x(i)^3 + x(i)^4;
        y2(i) = (1 / (10 + abs(x(i)))) + (1 / (1 + exp(-x(i))));
    end
    data(:, 1) = x;
    data(:, 2) = y1;
    data(:, 3) = y2;
    data = Util.scale(data, 2);
end