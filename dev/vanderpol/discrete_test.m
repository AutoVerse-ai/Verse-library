x0 = [1.4, 2.3];
dt = 0.01;

steps = 300;
for j=1:20
    t0 = 0;

    traj = x0;
    t = t0;
    for i=1:steps
        x = traj(end,:);
        x_plus = x;
        w1 = unifrnd(0.9, 1.1);
        w2 = unifrnd(0.9, 1.1);
%         syms w1 w2 real
        x_plus(1) = x(1) + dt*x(2)*w1;
        x_plus(2) = x(2) + dt*((1-x(1)^2)*x(2)*w2-x(1));
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
end


x10 = 1.4;
x20 = 2.3;
x10_hat = 1.4;
x20_hat = 2.3;
w1 = 0.9;
w1_hat = 1.1;
w2 = 0.9;
w2_hat = 1.1;
t0 = 0;
dt = 0.01;

traj = [x10, x20, x10_hat, x20_hat];
t = t0;
for i=1:steps
    disp(i)
    tmp = traj(end,:);
    x1 = tmp(1);
    x2 = tmp(2);
    x1_hat = tmp(3);
    x2_hat = tmp(4);
    [x1_plus, x2_plus] = computeDVanderpol([x1,x2],[w1, w2],[x1_hat,x2_hat],[w1_hat,w2_hat],dt);
    [x1_hat_plus, x2_hat_plus] = computeDVanderpol([x1_hat,x2_hat],[w1_hat,w2_hat],[x1,x2],[w1, w2],dt);
    t0 = t0+dt;
    traj = [traj;[x1_plus,x2_plus,x1_hat_plus,x2_hat_plus]];
    t = [t,t0];
end

figure(1)
plot(t, traj(:,1),'r')
plot(t, traj(:,3),'g')

figure(2)
plot(t, traj(:,2),'r')
plot(t, traj(:,4),'g')

