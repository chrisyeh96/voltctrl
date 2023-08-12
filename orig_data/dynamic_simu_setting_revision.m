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
