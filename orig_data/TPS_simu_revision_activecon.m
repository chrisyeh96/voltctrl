% Guannan Qu, 03/26/2019
% this file does simulation for Transaction for power systems, first
% revision. This is a replica of TPS_simu_revision.m to play with variation
% of paramters that create a scenario that many constrants are activated.
clear all;clc;close all;
rng('default');

% import network data
define_constants;
G = case56;
G.branch(G.branch(:,3)>1,3)=G.branch(G.branch(:,3)>1,3)/10;
% data in G are not p.u.
% input data to matpower are in p.u., except p, q are in MW (MVar)
G.branch(:,3:4) = G.branch(:,3:4)/(G.basekV)^2*G.baseMVA; % converting everything in G to be p.u.
G.bus(:,2:3) = G.bus(:,2:3)/G.baseMVA; % converting everything in G to be p.u.
PV_bus = 45;

[X,R] = generateRX(G.branch);  % get the X and R matrix
Y = inv(X);
n = size(X,1);


% case_specific settings

simu_case = 'static';

if(strcmp(simu_case,'static')==1)
    T=10000                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ;
    G.bus(19:end,2) = G.bus(19:end,2)*3; 
    G.bus(7:18,2) = G.bus(7:18,2)*2.8 ;
    PV_bus = [9,12,14,15,16,19,27,33,35,36,37,39,40,43,44,45,46,52]-1;
    G.bus(45,2) = 0;
    %G.bus(PV_bus,2) = -1;
    %G.bus(44:45,2) = -3;
    % for test cvx opf only!!!
    %G.bus(:,2:3)=0;
    % %%%%%%%%%%%%%%%%%%%%% 
  
    v_bar = 1.05^2; % upper limit for v
    v_un = 0.95^2; % lower limit for v
    v_bar_vec = v_bar*ones(n,1);
    v_un_vec  = v_un*ones(n,1);
    
    q_bar = +0.1;% upper limit for q
    q_un  = -0.1; % lower limit for q
    q_bar_vec = q_bar*ones(n,1);
    q_un_vec  = q_un*ones(n,1);
    
    % defining cost function
    c_price = 10; 
    s_max = rand(n,1)*0.5+0.5;
    a = zeros(n,1);
    a(PV_bus) = 2* c_price./s_max(PV_bus);%ones(N,1);%1+rand(N,1);% ones(N,1);
    b =  zeros(n,1);
    power_loss_weight = 1;
    
    case_mpc = convert2matpower(G);
    v_par = R*(-G.bus(2:end,2))+ X*(-G.bus(2:end,3))+1;
    
    volt = @(control_action) solve_v_matpower(case_mpc,control_action);
    no_control_voltage = volt(zeros(n,1));
    %volt = @(control_action) X*control_action+v_par;
    alpha = 0.00001;
    %beta = 5;
    %gamma = 1;
    beta = .3;
    gamma = 1;

%     cvx_begin
%     variable qq(n)
%     minimize 1/2*transpose(qq) *diag(a)*(qq )+ transpose(qq)*b + 1/2*power_loss_weight*transpose(qq)*X*qq
%     subject to
%     v_un <= X*qq+v_par <= v_bar
%     q_un <= qq <= q_bar
%     cvx_end
    
    [opt_val,opt_q_inj,opt_v]=cvx_OPF(G,v_un_vec,v_bar_vec,q_un_vec,q_bar_vec,X,a,b,power_loss_weight);
%    case_mpc_opf = convert2matpower_with_gencost(G,a,b);
%    opf_results = runopf(case_mpc_opf);

elseif(strcmp(simu_case, 'dynamic')==1)
    T = 14421;
    v_bar = 1.05^2; % upper limit for v
    v_un = 0.95^2; % lower limit for v
    v_bar_vec = v_bar*ones(n,1);
    v_un_vec  = v_un*ones(n,1);
    
    q_bar = +0.2;% upper limit for q
    q_un  = -0.2; % lower limit for q
    q_bar_vec = q_bar*ones(n,1);
    q_un_vec  = q_un*ones(n,1);
    
    v_par = R*(-G.bus(2:end,2)) + X*(-G.bus(2:end,3))+1;  %v_par; v= X*q + v_par;
   
    
    
    case_mpc = convert2matpower(G);
    [N_loads, ~,P_I,Q_I] = Generate_loads;
    pq_fluc = zeros(n,2,T);
    pq_fluc(:,1,:) = -P_I(:,1:T)/1000 *1.25;
    pq_fluc(:,2,:) = -Q_I(:,1:T)/1000 *1.25;
    pq_load_profile = pq_fluc;
    % generate fluctuating PV data
    %fluc_period = 30*10; % 30 min
    %fluc_time = ceil(T/fluc_period);
    %p_generation_vec = transpose(vec(repmat(1*rand(1,fluc_time),fluc_period,1))); 
    %p_generation_vec = repmat(p_generation_vec(1:T),length(PV_bus),1);
    
    % generate PV from real data
    load_PV_CX;
    PV_data = PV_3d_av(:,1:T)/1000*15; % change to MV
    PV_bus = [9,12,14,15,16,19,27,33,35,36,37,39,40,43,44,45,46,52]-1;
    for(k=1:length(PV_bus))
        temp_vec = zeros(1,1,T);
        temp_vec(1,1,:) = PV_data(k,:);
        pq_fluc(PV_bus(k),1,:) = pq_fluc(PV_bus(k),1,:) + temp_vec;
    end

    % defining cost function
    c_price = 10; 
    s_max = rand(n,1)*0.5+0.5;
    a = zeros(n,1);
    a(1:50) = 2* c_price./s_max(1:50);%ones(N,1);%1+rand(N,1);% ones(N,1);
    %a = a.*(randi(2,n,1)-1); % some of the a_i will be set as zero for zero cost function
    b =  zeros(n,1);
    power_loss_weight = 1;
    %
    volt = @(control_action,loading_profile) solve_v_matpower(case_mpc,control_action,loading_profile);
    alpha = 0.00001;
    beta = 0.5;
    gamma = 1;
    
    time_span = (1:T)/(10*60);
elseif(strcmp(simu_case , 'measurement')==1)
    dynamic_simu_setting_revision;
    noise = 0.03;
elseif(strcmp(simu_case , 'delay')==1)
    dynamic_simu_setting_revision;
    
    alpha = 0.0000035;
    beta = .5;
    gamma = 1;
    delay = 10;
    
elseif(strcmp(simu_case , 'model')==1)
    dynamic_simu_setting_revision;
    Y11 = sum(Y(1,:));
    Y_err = Y.*(rand(n,n)*0.4+0.8).*(ones(n,n) - eye(n)); 
    Y_err = Y_err - diag(sum(Y_err,2));
    Y_err(1,1) = Y_err(1,1)+Y11;
    %alpha = 0.00003;
end

%% run our algorithm
if strcmp(simu_case,'static')==1
    [v,q,~,f] = optdist_vc(T,[alpha,beta,gamma],v_un_vec,v_bar_vec,q_un_vec,q_bar_vec,volt,a,b,power_loss_weight,Y);
elseif(strcmp(simu_case,  'dynamic')==1)
    [v,q,~,f] = optdist_vc(T,[alpha,beta,gamma],v_un_vec,v_bar_vec,q_un_vec,q_bar_vec,volt,a,b,power_loss_weight,Y,pq_fluc);
    v_noncontrol = nocontrol(T,volt,pq_fluc);
elseif(strcmp(simu_case , 'measurement')==1)
        [v,q,~,f] = optdist_vc(T,[alpha,beta,gamma],v_un_vec,v_bar_vec,q_un_vec,q_bar_vec,volt,a,b,power_loss_weight,Y,pq_fluc,noise);
elseif(strcmp(simu_case , 'delay')==1)
        [v,q,~,f] = optdist_vc(T,[alpha,beta,gamma],v_un_vec,v_bar_vec,q_un_vec,q_bar_vec,volt,a,b,power_loss_weight,Y,pq_fluc,0,delay);
elseif(strcmp(simu_case , 'model')==1)
        [v,q,~,f] = optdist_vc(T,[alpha,beta,gamma],v_un_vec,v_bar_vec,q_un_vec,q_bar_vec,volt,a,b,power_loss_weight,Y_err,pq_fluc);

end
%% 
font_size=14;
if strcmp(simu_case , 'static')==1
    bus_to_display_plus_one = [10,19,36,37,39,41,46,47,55,56];
    legend_txt = generate_legend_txt(bus_to_display_plus_one);
    bus_to_display = bus_to_display_plus_one - 1;
        figure;
    %subplot(3,1,1);
    plot(1:T,sqrt(v(bus_to_display,:)')*G.basekV,'LineWidth',1.5);hold on;
    plot(1:T,sqrt(v_bar)*G.basekV*ones(1,T),'--','LineWidth',0.7);
    plot(1:T,sqrt(v_un)*G.basekV*ones(1,T),'--','LineWidth',0.7);
    legend(legend_txt);
    
    xlabel('Iterations');
    ylabel('Voltage (kV)');
    title('Voltage Profile');
    ylim([10.5,13]);
        set(gca,'FontSize',font_size);

    
    figure;
    %subplot(3,1,2);
    plot(1:T,q(bus_to_display,:)','LineWidth',1.5);hold on;
    plot(1:T,q_bar*ones(1,T),'--','LineWidth',0.7);
    plot(1:T,q_un*ones(1,T),'--','LineWidth',0.7);

    xlabel('Iterations');

    ylabel('Reactive Power (MVar)');
    title('Reactive power injection');
        legend(legend_txt);

    ylim([-0.3,0.3]);
        set(gca,'FontSize',font_size);

        figure;
    plot(1:T,f,'LineWidth',1.5);hold on;
    plot(1:T,opt_val*ones(1,T),'--','LineWidth',1.5);
    xlabel('Iterations');
    ylabel('Cost');
    title('Cost Function');
    set(gca,'FontSize',font_size);

elseif(strcmp(simu_case , 'dynamic')==1)
    bus_to_display_plus_one = [9,19,22,31,40,46,55];
    legend_txt = generate_legend_txt(bus_to_display_plus_one);
    bus_to_display = bus_to_display_plus_one - 1;
    figure;
    plot(v');hold on;
    plot(1:T,v_bar*ones(1,T),'--','LineWidth',0.7);
    plot(1:T,v_un*ones(1,T),'--','LineWidth',0.7);
    figure;
    plot(v_noncontrol');hold on;
    plot(1:T,v_bar*ones(1,T),'--','LineWidth',0.7);
    plot(1:T,v_un*ones(1,T),'--','LineWidth',0.7);
    figure;
    %subplot(3,1,1);
    plot(time_span,sqrt(v(bus_to_display,:)')*G.basekV,'LineWidth',1.5);hold on;
    plot(time_span,sqrt(v_bar)*G.basekV*ones(1,T),'--','LineWidth',0.7);
    plot(time_span,sqrt(v_un)*G.basekV*ones(1,T),'--','LineWidth',0.7);
    legend(legend_txt);
    set(gca,'FontSize',font_size);
    
    xlabel('Time (hour)');
    ylabel('Voltage (kV)');
    title('Voltage Profile');
    xlim([0,24]);
    ylim([10.5,13]);
    set(gca,'XTick',[0,4,8,12,16,20,24]);
    set(gca,'XTickLabel',{'00:00','04:00','08:00','12:00','16:00','20:00','24:00'});
    
    
    figure;
    %subplot(3,1,2);
    plot(time_span,q(bus_to_display,:)','LineWidth',1.5);hold on;
        plot(time_span,q_bar*ones(1,T),'--','LineWidth',0.7);
    plot(time_span,q_un*ones(1,T),'--','LineWidth',0.7);

    xlim([0,24]);
    set(gca,'XTick',[0,4,8,12,16,20,24]);
    set(gca,'XTickLabel',{'00:00','04:00','08:00','12:00','16:00','20:00','24:00'});
        xlim([0,24]);
    xlabel('Time (hour)');

    ylabel('Reactive Power (MVar)');
    title('Reactive power injection');
        legend(legend_txt);
    set(gca,'FontSize',font_size);

    ylim([-0.3,0.3]);
        figure;
    %subplot(3,1,1);
    plot(time_span,sqrt(v_noncontrol(bus_to_display,:)')*G.basekV,'LineWidth',1.5);hold on;
    plot(1:T,sqrt(v_bar)*G.basekV*ones(1,T),'--','LineWidth',0.7);
    plot(1:T,sqrt(v_un)*G.basekV*ones(1,T),'--','LineWidth',0.7);
    legend(legend_txt);
    set(gca,'FontSize',font_size);
    
    xlabel('Time (hour)');
    ylabel('Voltage (kV)');
    title('Voltage Profile without Controller');
    xlim([0,24]);
    set(gca,'XTick',[0,4,8,12,16,20,24]);
    set(gca,'XTickLabel',{'00:00','04:00','08:00','12:00','16:00','20:00','24:00'});
        legend(legend_txt);
    set(gca,'FontSize',font_size);

    
    %figure;
  %  plot(time_span,q_
    
   % subplot(3,1,3)
   figure;
    plot(time_span,f,'LineWidth',2);
        xlim([0,24]);
    set(gca,'XTick',[0,4,8,12,16,20,24]);
    set(gca,'XTickLabel',{'00:00','04:00','08:00','12:00','16:00','20:00','24:00'});
    
elseif strcmp(simu_case , 'measurement')==1
    bus_to_display_plus_one = [9,19,22,31,40,46,55];
    legend_txt = generate_legend_txt(bus_to_display_plus_one);
    bus_to_display = bus_to_display_plus_one - 1;
    
    figure;
    %subplot(3,1,1);
    plot(time_span,sqrt(v(bus_to_display,:)')*G.basekV,'LineWidth',1.5);hold on;
    plot(time_span,sqrt(v_bar)*G.basekV*ones(1,T),'--','LineWidth',0.7);
    plot(time_span,sqrt(v_un)*G.basekV*ones(1,T),'--','LineWidth',0.7);
    legend(legend_txt);
    set(gca,'FontSize',font_size);
    
    xlabel('Time (hour)');
    ylabel('Voltage (kV)');
    title('Measurement Noise and Delay');
    xlim([0,24]);
    ylim([10.5,13]);
    set(gca,'XTick',[0,4,8,12,16,20,24]);
    set(gca,'XTickLabel',{'00:00','04:00','08:00','12:00','16:00','20:00','24:00'});

elseif strcmp(simu_case , 'delay')==1
    bus_to_display_plus_one = [9,19,22,31,40,46,55];
    legend_txt = generate_legend_txt(bus_to_display_plus_one);
    bus_to_display = bus_to_display_plus_one - 1;
    
    figure;
    %subplot(3,1,1);
    plot(time_span,sqrt(v(bus_to_display,:)')*G.basekV,'LineWidth',1.5);hold on;
    plot(time_span,sqrt(v_bar)*G.basekV*ones(1,T),'--','LineWidth',0.7);
    plot(time_span,sqrt(v_un)*G.basekV*ones(1,T),'--','LineWidth',0.7);
    legend(legend_txt);
    set(gca,'FontSize',font_size);
    
    xlabel('Time (hour)');
    ylabel('Voltage (kV)');
    title('Communication Delay');
    xlim([0,24]);
    ylim([10.5,13]);
    set(gca,'XTick',[0,4,8,12,16,20,24]);
    set(gca,'XTickLabel',{'00:00','04:00','08:00','12:00','16:00','20:00','24:00'});

elseif strcmp(simu_case , 'model')==1
    bus_to_display_plus_one = [9,19,22,31,40,46,55];
    legend_txt = generate_legend_txt(bus_to_display_plus_one);
    bus_to_display = bus_to_display_plus_one - 1;
    
    figure;
    %subplot(3,1,1);
    plot(time_span,sqrt(v(bus_to_display,:)')*G.basekV,'LineWidth',1.5);hold on;
    plot(time_span,sqrt(v_bar)*G.basekV*ones(1,T),'--','LineWidth',0.7);
    plot(time_span,sqrt(v_un)*G.basekV*ones(1,T),'--','LineWidth',0.7);
    legend(legend_txt);
    set(gca,'FontSize',font_size);
    
    xlabel('Time (hour)');
    ylabel('Voltage (kV)');
    title('Modeling Error');
    xlim([0,24]);
    ylim([10.5,13]);
    set(gca,'XTick',[0,4,8,12,16,20,24]);
    set(gca,'XTickLabel',{'00:00','04:00','08:00','12:00','16:00','20:00','24:00'});
end
%% plot load and PV generation
figure;
plot(time_span,-squeeze(sum(pq_load_profile(:,1,:),1)),'LineWidth',1.5); hold on;
plot(time_span,-squeeze(sum(pq_load_profile(:,2,:),1)),'LineWidth',1.5);
plot(time_span,sum(PV_data,1),'LineWidth',1.5);
legend({'Active Load','Reactive Load', 'PV Generation'},'Location','NorthWest');
xlim([0,24]);
set(gca,'XTick',[0,4,8,12,16,20,24]);
set(gca,'XTickLabel',{'00:00','04:00','08:00','12:00','16:00','20:00','24:00'});
xlabel('Time (hour)');
ylabel('MW (MVar)');
title('Aggregate Load and PV Generation Profile');
set(gca,'FontSize',font_size);