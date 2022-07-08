syms x1 x2 w1 w2 real
x = [x1,x2];
f = -x2*(1-x1^2);
fh = matlabFunction(f,'vars',{x});
% fh = @(x)x(2)*(1-x(1)^2);
% fh = @(x)x(2)*(1-x(1)^2);

[x,fval] = fmincon(fh,[1,1],[],[],[],[],[-2.5,-3],[2.5,3])

% options = optimoptions('fminunc','Display','final','Algorithm','quasi-newton');
% fh2 = matlabFunction(f,'vars',{x}); 
% % fh2 = objective with no gradient or Hessian
% [xfinal,fval,exitflag,output2] = fminunc(fh2,[-1;2],[],[],[],[],[],[],constraint,options)
