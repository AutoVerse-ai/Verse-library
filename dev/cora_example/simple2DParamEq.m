function dx = simple2DParamEq(x,u,w)
    dx(1,1) = x(1)*(1.1+w(1)-x(1)-0.1*x(2)); 
    dx(2,1) = x(2)*(4+w(2)-3*x(1)-x(2));
end