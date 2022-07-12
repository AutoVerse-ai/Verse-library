function d = computeDecomposition(exprs, symbol_x, symbol_w, x, w, x_hat, w_hat, dt)
%COMPUTEDECOMPOSITION Summary of this function goes here
%   Detailed explanation goes here
    d = zeros(size(exprs));
    for j=1:length(d)
        expr = symbol_x(j) + exprs(j)*dt;
        if all(x<=x_hat) && all(w<=w_hat)
            var_range = containers.Map;
            for i=1:length(symbol_x)
                var_range(char(symbol_x(i))) = [x(i), x_hat(i)];
            end
            for i=1:length(symbol_w)
                var_range(char(symbol_w(i))) = [w(i), w_hat(i)];
            end
            d(j) = find_min(expr, var_range);
        elseif all(x>=x_hat) && all(w>=w_hat)
            var_range = containers.Map;
            for i=1:length(symbol_x)
                var_range(char(symbol_x(i))) = [x_hat(i), x(i)];
            end
            for i=1:length(symbol_w)
                var_range(char(symbol_w(i))) = [w_hat(i), w(i)];
            end
            d(j) = find_max(expr, var_range);
        else
        end
    end
end

