x0 = [1, 1.5, 1, 0];
dt = 0.01;

steps = 1000;
for j=1:20
    t0 = 0;

    tmp = x0;
    tmp(1) = unifrnd(1, 1.5);
    tmp(2) = unifrnd(1, 1.5);
    traj = tmp;
    t = t0;
    for i=1:steps
        x = traj(end,:);
        x_plus = x;
        w = unifrnd(-0.1, 0.1);
%         syms w1 w2 real
        x_plus(1) = x(1) + dt*(-2*x(1) + x(2)*(1 + x(1)) + x(3) + w);
        x_plus(2) = x(2) + dt*(-x(2) + x(1)*(1 - x(2)) + 0.1);
        x_plus(3) = x(3) + dt*(-x(4));
        x_plus(4) = x(4) + dt*(x(3));
        var_range = containers.Map;
%         var_range(char(w1))=[0.9,1.1];
%         var_range(char(w2))=[0.9,1.1];
        
%         x_plus(1) = find_min(expr1, var_range);
%         x_plus(2) = find_max(expr2, var_range);
        t0 = t0+dt;
        traj = [traj;x_plus];
        t = [t,t0];
    end

    figure(1)
    plot(t, traj(:,1),'b')
    hold on

    figure(2)
    plot(t, traj(:,2),'b')
    hold on

    figure(3)
    plot(traj(:,1), traj(:,2), 'b')
    hold on
end

x10 = 1;
x20 = 1;
x30 = 1;
x40 = 0;
x10_hat = 1.5;
x20_hat = 1.5;
x30_hat = 1;
x40_hat = 0;
t0 = 0;

traj = [x10, x20, x30, x40, x10_hat, x20_hat, x30_hat, x40_hat];
t = t0;
syms x1 x2 x3 x4 w real
exprs = [-2*(x1)+x2*(1+x1)+x3+w,-x2+x1*(1-x2)+0.1,-x4,x3];
for i=1:steps
    disp(i)
    tmp = traj(end,:);
    d = computeDecomposition(exprs,[x1,x2,x3,x4],[w],tmp(1:4),[-0.1],tmp(5:8),[0.1],dt);
    d_hat = computeDecomposition(exprs,[x1,x2,x3,x4],[w],tmp(5:8),[0.1],tmp(1:4),[-0.1],dt);
    t0 = t0+dt;
    traj = [traj;[d, d_hat]];
    t = [t,t0];
end

figure(1)
plot(t, traj(:,1),'r')
plot(t, traj(:,5),'g')

figure(2)
plot(t, traj(:,2),'r')
plot(t, traj(:,6),'g')

figure(3)
for i=1:size(traj,1)
    rectangle('Position',[traj(i,1), traj(i,2), traj(i,5)-traj(i,1), traj(i,6)-traj(i,2)], 'EdgeColor', 'g')
end
