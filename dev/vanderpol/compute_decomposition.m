syms x1 x2 w1 w2 real

x = [x1,x2];
w = [w1,w2];
f = [w1*x2,(1-x1^2)*x2*w2-x1];
var_range = containers.Map;
var_range(char(x1))=[1,2];
var_range(char(x2))=[1,3];
var_range(char(w1))=[sym('9/10'),sym('11/10')];
var_range(char(w2))=[sym('9/10'),sym('11/10')];

% Compute jx and jw
jx = jacobian(f,x);
jw = jacobian(f,w);

% Compute jx_lower and jx_upper
jx_lower = sym(round(find_min(jx, var_range),5));
jx_upper = sym(round(find_max(jx, var_range),5));

jw_lower = sym(round(find_min(jw, var_range),5));
jw_upper = sym(round(find_max(jw, var_range),5));

% Determine delta and epsilon
delta = [0,1;0,0];
epsilon = [1,0;0,0];

x_hat = x;
w_hat = w;
for i=1:length(x_hat)
    tmp = [char(x_hat(i)),'_hat'];
    x_hat(i) = sym(tmp);
    assume(x_hat(i),'real')
end

for i=1:length(w_hat)
    tmp = [char(w_hat(i)),'_hat'];
    w_hat(i) = sym(tmp);
    assume(w_hat(i),'real')
end

% Determine zeta alpha pi and beta
zeta = jx;
alpha = zeros(size(jx));
zeta_size = size(zeta);
for i=1:zeta_size(1)
    for j=1:zeta_size(2)
        if i==j
            zeta(i,j) = x(i);
            alpha(i,j) = 0;
        else
            if delta(i,j)==1
                zeta(i,j) = x_hat(j);
                alpha(i,j) = jx_upper(i,j);
            else
                zeta(i,j) = x(j);
                alpha(i,j) = -jx_lower(i,j);
            end
        end
    end
end

pi = jw;
beta = zeros(size(jw));
pi_size = size(pi);
for i=1:pi_size(1)
    for k=1:pi_size(2)
        if epsilon(i,k)==0
            pi(i,k) = w(k);
            beta(i,k) = -jw_lower(i,k);
        else
            pi(i,k) = w_hat(k);
            beta(i,k) = jw_upper(i,k);
        end
    end
end

% Substitute zeta, alpha, pi and beta
d = f;
for i=1:length(d)
    d(i) = subs(f(i), [x,w], [zeta(i,:),pi(i,:)])+alpha(i,:)*(x-x_hat)'+beta(i,:)*(w-w_hat)';
end

% Check if d is a valid decomposition
% Update var_range with the _hat variables
for i=1:length(x)
    tmp = char(x(i));
    bound = var_range(tmp);
    var_range([tmp,'_hat']) = bound;
end
for i=1:length(w)
    tmp = char(w(i));
    bound = var_range(tmp);
    var_range([tmp,'_hat']) = bound;
end

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


if ~res
    disp 'invalid decomposition'
    return
end

d_hat = d;
for i=1:length(d_hat)
    d_hat(i) = subs(d(i), [x,w,x_hat, w_hat],[x_hat, w_hat, x,w]);
end

% Substitute \underline{w} and \overline{w}
w_lower = zeros(size(w));
w_upper = w_lower;
for i=1:length(w)
    symb = w(i);
    bound = var_range(char(symb));
    w_lower(i) = bound(1);
    w_upper(i) = bound(2);
end

e = d;
for i=1:length(e)
    e(i) = subs(d(i), [w, w_hat], [w_lower, w_upper]);
end

e_hat = d_hat;
for i=1:length(e_hat)
    e_hat(i) = subs(d_hat(i), [w, w_hat], [w_lower, w_upper]);
end

e = [e, e_hat];

vpa(d', 4)
vpa([d,d_hat]', 4)
vpa(e', 4)