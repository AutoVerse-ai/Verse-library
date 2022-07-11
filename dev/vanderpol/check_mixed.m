function res = check_mixed(d, x, x_hat, w, w_hat, var_range)
    res = true;
    % Check increasing with x
    for i=1:length(d)
        di=d(i);
        for j=1:length(x)
            if i~=j
                dddx = diff(di, x(j));
                val = find_min(dddx,var_range);
                res = res&(val>=0);
            end
        end
    end
    
    % Check decreasing with x_hat
    for i=1:length(d)
        di=d(i);
        for j=1:length(x_hat)
            dddx = diff(di, x_hat(j));
            val = find_max(dddx, var_range);
            res = res&(val<=0);
        end
    end
    
    % Check increasing with w
    for i=1:length(d)
        di=d(i);
        for j=1:length(w)
            dddx=diff(di, w(j));
            val=find_min(dddx,var_range);
            res = res&(val>=0);
        end
    end
    
    % Check decreasing with w_hat
    for i=1:length(d)
        di=d(i);
        for j=1:length(w_hat)
            dddx=diff(di, w_hat(j));
            val=find_max(dddx,var_range);
            res = res&(val<=0);
        end
    end
end

