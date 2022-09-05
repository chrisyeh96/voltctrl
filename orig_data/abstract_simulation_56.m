clear all;close all;
rng('default');
G = case56;
G.branch = G.branch(1:14,:);
[X,R] = generateRX(G.branch); Y = inv(X);
n = size(X,1);
% P = G.bus(2:end,2);
P = zeros(n,1);
v_par = R*P+1;% v_par; v= X*q + v_par;

q_bar = zeros(n,1)+.100;% upper limit for q
q_un = zeros(n,1) - .100; % lower limit for q

v_bar = 1.05^2; % upper limit for v
v_un = 0.95^2; % lower limit for v

cost_weight = rand(n,1)+1; % reactive power cost = 1/2*cost_weight*(q - cost_offset)^2
cost_offset = 2*rand(n,1)-1;% reactive power cost 1/2*cost_weight*(q - cost_offset)^2

power_loss_weight = 1; % cost = reactive power cost + power_loss_weight*1/2*q'Xq
nabla = @(qq) diag(cost_weight)*(qq - cost_offset) + power_loss_weight* X*qq;
f_func = @(qq) 1/2*(qq - cost_offset)'*diag(cost_weight)*(qq - cost_offset) + 1/2*power_loss_weight*qq'*X*qq;

T = 500; % number of steps to simulate
q_hat = zeros(n,T); % ''virtual'' reactive power
xi = zeros(n,T); % lagrangian multiplier for reactive power constraint
lambda_bar = zeros(n,T); % lagrangian multipler for voltage constraint (upper limit)
lambda_un = zeros(n,T); % lagrangian multipler for voltage constraint (lower limit)
v = zeros(n,T); % voltage
q = zeros(n,T); % ''actual'' reactive power
f = zeros(1,T); % objective function value

% step sizes
alpha = 0.05;
beta = 0.1;
gamma = 0.1;

c = 1;% parameter for augmented Lagrangian
K = 10;% number of inner loop iterations
for t=1:T
    % setting tilde_q(t,0)
    if(t == 1)
        tilde_q = zeros(n,1);
    else
        tilde_q = q_hat(:,t-1);
    end
    for k=1:K
        % gradient of the quadratic penalty function, evaluated at xi(:,t),  tilde_q
        G = max(min(xi(:,t) + c*(tilde_q - q_un),0), xi(:,t)+c*(tilde_q - q_bar));  
        % gradient descent step for tilde_q
        tilde_q = tilde_q - alpha*(lambda_bar(:,t) - lambda_un(:,t) + Y*(nabla(tilde_q) + G ));
    end
    q_hat(:,t) = tilde_q; % setting virtual reactive power to be the last iteration of tilde_q
    %q_hat(:,t) = X*(lambda_un(:,t) - lambda_bar(:,t)) + xi_un(:,t) - xi_bar(:,t);
    q(:,t) = max(min(q_hat(:,t),q_bar),q_un); % ''actuall implemented reactive power''
    v(:,t) = X*q(:,t)+v_par; % voltage
    f(t) = f_func(q(:,t));
    if(t<T)
        % update for the multipliers
        for i=1:n
            if q_hat(i,t) + xi(i,t)/c<q_un(i)
                xi(i,t+1) = xi(i,t) + beta*(q_hat(i,t) - q_un(i));
            elseif q_hat(i,t) + xi(i,t)/c > q_bar(i)
                xi(i,t+1) = xi(i,t) + beta*(q_hat(i,t) - q_bar(i));
            else
                xi(i,t+1) = (1 - beta/c)*xi(i,t);
            end
        end
        %xi_bar(:,t+1) = max(xi_bar(:,t) + beta * (q_hat(:,t) - q_bar), 0);
        %xi_un(:,t+1)  = max(xi_un(:,t) + beta * (q_un - q_hat(:,t)),0);
        lambda_bar(:,t+1) = max(lambda_bar(:,t) + gamma* (v(:,t) - v_bar),0);
        lambda_un(:,t+1) = max(lambda_un(:,t) + gamma*(v_un - v(:,t)),0);
    end
end

%% Solving centralized solution
% centralized optimization problem - start
cvx_begin
variable qq(n)
minimize (1/2*(qq - cost_offset)'*diag(cost_weight)*(qq - cost_offset) + 1/2*power_loss_weight*qq'*X*qq)
subject to 
v_un <= X*qq+v_par <= v_bar
q_un <= qq <= q_bar
cvx_end


% centralized optimization problem - end

%% plots
v_sqroot = sqrt(v); 

fontsize = 15;
figure;
subplot(3,1,1);
plot(v_sqroot(7:9,:)','LineWidth',2);hold on;
plot(1:T,sqrt(v_bar)*ones(1,T),'--','LineWidth',0.7);
plot(1:T,sqrt(v_un)*ones(1,T),'--','LineWidth',0.7);
title('Voltage Profile','FontSize',fontsize); 
ylim([0.7,1.2]);
set(gca,'FontSize',fontsize);

subplot(3,1,2);
plot(q(7:9,:)','LineWidth',2);hold on;
plot(1:T,q_bar(1)*ones(1,T),'--','LineWidth',0.7);
plot(1:T,q_un(1)*ones(1,T),'--','LineWidth',0.7);
ylim([-0.2,0.2]);
title('Reactive Power Injection','FontSize',fontsize);
set(gca,'FontSize',fontsize);

subplot(3,1,3);
plot(f, 'LineWidth',2); hold on;
plot(1:T,cvx_optval*ones(1,T),'-.','LineWidth',0.7); 
title('Cost Function','FontSize',fontsize);
xlabel('Iterations','FontSize',fontsize);
set(gca,'FontSize',fontsize);

% figure;
% plot(lambda_bar'-lambda_un');title('\lambda');
% 
% 
% figure;
% plot(v'); title('v');
% 
% figure; 
% plot(q_hat'); title('q_{hat}');
% 
% figure;
% plot(xi_bar(1,:),'b'); hold on;
% plot(xi_un(1,:),'r'); hold on;