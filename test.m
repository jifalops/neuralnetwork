function [a b c] = test(x, y, z)
    a = x^2
    b = y^2
    throw(MException('VerifyOutput:OutOfBounds', ...
                    'Cannot train without input/output data.'));
    c = z^2
end