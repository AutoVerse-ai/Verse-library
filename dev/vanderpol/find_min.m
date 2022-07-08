function res = find_min(exprs,var_range)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    res = zeros(size(exprs));
    res_size = size(res);
    for i=1:res_size(1)
        for j=1:res_size(2)
            f = exprs(i,j);
            symbols = symvar(f);
            if isempty(symbols)
                res(i,j) = double(f);
                continue
            end
            fh = matlabFunction(f,'vars',{symbols});
            
            lb = zeros(size(symbols));
            ub = zeros(size(symbols));
            for k=1:length(symbols)
                bound = var_range(char(symbols(k)));  
                lb(k) = bound(1);
                ub(k) = bound(2);
            end
            [x,fval] = fmincon(fh,ub,[],[],[],[],lb,ub);
            res(i,j) = fval;
        end
    end
end

