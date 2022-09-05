clear all;close all;
rng('default');
%%  ======================= load and process network data  =======================
define_constants;
G = case56;
G.branch(G.branch(:,3)>1,3)=G.branch(G.branch(:,3)>1,3)/10;
% G.branch = G.branch(1:14,:);
% data in G are not p.u.
% input data to matpower are in p.u., except p, q are in MW (MVar)
G.branch(:,3:4) = G.branch(:,3:4)/(G.basekV)^2*G.baseMVA; % converting everything in G to be p.u.
G.bus(:,2:3) = G.bus(:,2:3)/G.baseMVA; % converting everything in G to be p.u.
PV_bus = 45;

[X,R] = generateRX(G.branch);  % get the X and R matrix
Y = inv(X);
n = size(X,1);
simu_case = 3; % 1 - moderate load, large PV; 2 - heavy load, large PV; 3 - fluctuating generation
rng('default');
%%  ============================ Case Spedific Parameters ===========================

T = 500; % number of steps to simulate

if(simu_case == 1)
    q_bar = zeros(n,1)+1;% upper limit for q
    q_un = zeros(n,1) - 1; % lower limit for q
    G.bus(:,2) =G.bus(:,2)/10;
    G.bus(PV_bus,2) = -7;
    case_mpc = convert2matpower(G);
    v_par = R*(-G.bus(2:end,2)) + X*(-G.bus(2:end,3))+1; % v_par for DistLineFlow Model
    
    cost_weight = rand(n,1); % reactive power cost = 1/2*cost_weight*(q - cost_offset)^2
    cost_offset = 4*(rand(n,1)-0.5); % reactive power cost 1/2*cost_weight*(q - cost_offset)^2
    
    % step sizes
    alpha = 0.0002;
    beta = 1;
    gamma = 100;
    
elseif(simu_case == 2)
    %    G.bus(:,2) = 3*G.bus(:,2);
    G.bus(19:end,2)=G.bus(19:end,2)/100;
    G.bus(7:18,2) = 6.5*G.bus(7:18,2) ;
    G.bus(PV_bus,2) = -8.5;
    
    q_bar = zeros(n,1)+1;% upper limit for q
    q_un  = zeros(n,1)-1; % lower limit for q
    
    case_mpc = convert2matpower(G);
    v_par = R*(-G.bus(2:end,2))+ X*(-G.bus(2:end,3))+1;
    
    cost_weight = rand(n,1); % reactive power cost = 1/2*cost_weight*(q - cost_offset)^2
    cost_offset =  4*(rand(n,1)-0.5); % reactive power cost 1/2*cost_weight*(q - cost_offset)^2
    % step sizes
    alpha = 0.0001;
    beta = 1;
    gamma = 100;
    
elseif(simu_case == 3)
    G.bus(:,2) =3*G.bus(:,2);
    G.bus(PV_bus,2) = -5;
    
    q_bar = zeros(n,1)+1;% upper limit for q
    q_un  = zeros(n,1)-1; % lower limit for q
    
    v_par = R*(-G.bus(2:end,2)) + X*(-G.bus(2:end,3))+1;  %v_par; v= X*q + v_par;
    
    cost_weight = rand(n,1); % reactive power cost = 1/2*cost_weight*(q - cost_offset)^2
    cost_offset = 4*(rand(n,1)-0.5); % reactive power cost 1/2*cost_weight*(q - cost_offset)^2
    
    % step sizes
    alpha = 0.0003;
    beta = 1;
    gamma = 100;
    
    flu_interval = 20;
end



%% ====================  Other Parameters & Initialize ====================

v_bar = 1.05^2; % upper limit for v
v_un = 0.95^2; % lower limit for v


power_loss_weight = 0; % cost = reactive power cost + power_loss_weight*1/2*q'Xq
nabla = @(qq) diag(cost_weight)*(qq - cost_offset) + power_loss_weight* X*qq;
f_func = @(qq) 1/2*(qq - cost_offset)'*diag(cost_weight)*(qq - cost_offset) + 1/2*power_loss_weight*qq'*X*qq;


% declaring variables

q_hat = zeros(n,T); % ''virtual'' reactive power
xi = zeros(n,T); % lagrangian multiplier for reactive power constraint
lambda_bar = zeros(n,T); % lagrangian multipler for voltage constraint (upper limit)
lambda_un = zeros(n,T); % lagrangian multipler for voltage constraint (lower limit)
v = zeros(n,T); % voltage
q = zeros(n,T); % ''actual'' reactive power
f = zeros(1,T); % objective function value


c = 1;% parameter for augmented Lagrangian

%% Run Algorithm
for t=1:T
    
    if(simu_case == 3)
        % fluctuating PV
        if(mod(t,flu_interval) ==1)
            G_fuctuating = G;
            G_fuctuating.bus(:,2) = G.bus(:,2).*(0.5+2.5*rand(n+1,1));
            case_mpc = convert2matpower(G_fuctuating);
            
        end
    end
    if(t>1)
        GG = max(min(xi(:,t) + c*(q_hat(:,t-1) - q_un),0), xi(:,t)+c*(q_hat(:,t-1) - q_bar));
        q_hat(:,t) = q_hat(:,t-1) - alpha*(lambda_bar(:,t) - lambda_un(:,t) + Y*(nabla(q_hat(:,t-1)) + GG ));
    end
    %q_hat(:,t) = X*(lambda_un(:,t) - lambda_bar(:,t)) + xi_un(:,t) - xi_bar(:,t);
    q(:,t) = max(min(q_hat(:,t),q_bar),q_un); % ''actuall implemented reactive power''
    
    % ================ Linear Model for Voltage ====================
    % ================ Non-Linear Model for Voltage =================
    tmp_case_mpc = case_mpc;
    tmp_case_mpc.bus(2:end,QD) = tmp_case_mpc.bus(2:end,QD) - q(:,t)*G.baseMVA; % bus injection
    pfresult = runpf(tmp_case_mpc);
    v(:,t) = (pfresult.bus(2:end,VM)).^2;
    %V_inj_1a(:,t) = V_1a(ind_inj_bus,t);
    
    
    
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

if(simu_case == 3)
    % squeeze matrix
   % v = v(:,flu_interval:flu_interval:T);
   % q = q(:,flu_interval:flu_interval:T);
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
v_sqroot =sqrt(v)*G.basekV; % sqrtroot of v
% if(simu_case<3)
%     number_of_plots = 3;
% else
%     number_of_plots=2;
% end

fontsize = 15;
figure;
if(simu_case == 1)
    selected_buses_plusone = [4,19,41,42,45];
    selected_buses = selected_buses_plusone - 1;
elseif(simu_case == 2)
    selected_buses_plusone = [4,19,41,42,45];
    selected_buses = selected_buses_plusone - 1;
    
    %selected_buses = 1:n;
else
    selected_buses_plusone = [4,25,34,45,53];
    selected_buses = selected_buses_plusone - 1;

end
num_bus_display = length(selected_buses_plusone);
legend_txt = cell(num_bus_display,1);
for ll = 1:num_bus_display
    legend_txt{ll} = ['bus ',int2str(selected_buses_plusone(ll))];
end

%selected_buses = 1:n;
subplot(3,1,1);
plot(v_sqroot(selected_buses,:)','LineWidth',2);hold on;
plot(1:size(v,2),sqrt(v_bar)*ones(1,size(v,2))*G.basekV,'--','LineWidth',0.7);
plot(1:size(v,2),sqrt(v_un)*ones(1,size(v,2))*G.basekV,'--','LineWidth',0.7);
title('Voltage Profile (kV)','FontSize',fontsize);
xlim([1,size(v,2)]);
ylim([0.8*G.basekV,1.1*G.basekV]);
set(gca,'FontSize',fontsize);
legend(legend_txt);

subplot(3,1,2);
plot(q(selected_buses,:)','LineWidth',2);hold on;
plot(1:size(v,2),q_bar(1)*ones(1,size(v,2)),'--','LineWidth',0.7);
plot(1:size(v,2),q_un(1)*ones(1,size(v,2)),'--','LineWidth',0.7);
xlim([1,size(v,2)]);
ylim([q_un(1)-0.5,q_bar(1)+0.5]);

%ylim([-1.5,0.5]);
title('Reactive Power Injection (MVar)','FontSize',fontsize);
set(gca,'FontSize',fontsize);

    subplot(3,1,3);
    plot(f, 'LineWidth',2); hold on;
    if(simu_case~=3)

        plot(1:size(v,2),cvx_optval*ones(1,size(v,2)),'-.','LineWidth',0.7);
    end
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