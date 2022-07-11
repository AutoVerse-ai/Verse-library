function [d1, d2] = computeDVanderpol(x, w, x_hat, w_hat, dt)
    syms x1 x2 w1 w2 real
    expr1 = x1+dt*(w1*x2);
    expr2 = x2+dt*((1-x1^2)*x2*w2-x1);
    if all(x<=x_hat) && all(w<=w_hat)
        var_range = containers.Map;
        var_range(char(x1))=[x(1),x_hat(1)];
        var_range(char(x2))=[x(2),x_hat(2)];
        var_range(char(w1))=[w(1),w_hat(1)];
        var_range(char(w2))=[w(2),w_hat(2)];
    
        d1 = find_min(expr1, var_range);
        d2 = find_min(expr2, var_range);
    elseif all(x>=x_hat) && all(w>=w_hat)
        var_range = containers.Map;
        var_range(char(x1))=[x_hat(1),x(1)];
        var_range(char(x2))=[x_hat(2),x(2)];
        var_range(char(w1))=[w_hat(1),w(1)];
        var_range(char(w2))=[w_hat(2),w(2)];
    
        d1 = find_max(expr1, var_range);
        d2 = find_max(expr2, var_range);

    else
        disp "Error!!!"
    end
end
