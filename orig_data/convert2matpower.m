function new_case =  convert2matpower(old_case)
define_constants;
new_case.version = '2';
new_case.baseMVA = old_case.baseMVA; 
N_bus = size(old_case.bus,1);
new_case.bus=zeros(N_bus,13);
for n=1:N_bus
   new_case.bus(n,BUS_I) = n;  % column 1
   if(n == 1)
       new_case.bus(n,BUS_TYPE) = 3; % Ref
   else
       new_case.bus(n,BUS_TYPE) = 1; % PQ bus
   end       % column 2
   new_case.bus(n,PD) = old_case.bus(n,2)*old_case.baseMVA; % column 3 
   new_case.bus(n,QD) = old_case.bus(n,3)*old_case.baseMVA; % old_case.bus(n,3); % column 4
   new_case.bus(n,GS) = 0; % column 5 
   new_case.bus(n,BS) = 0; % column 6 
   new_case.bus(n,BUS_AREA) = 1; % column 7 
   new_case.bus(n,VM) = 1; % column 8
   new_case.bus(n,VA) = 0; % column 9
   new_case.bus(n,BASE_KV) = old_case.basekV; % column 10
   new_case.bus(n,ZONE) = 1; % column 11; 
   new_case.bus(n,VMAX) = 1.05; % column 12;
   new_case.bus(n,VMIN) = 0.95; % column 13
   
end

N_branch = size(old_case.branch, 1);

new_case.branch = zeros(N_branch, 13);

new_case.branch(:,F_BUS) = old_case.branch(:,1);% column 1
new_case.branch(:,T_BUS) = old_case.branch(:,2);% column 2
new_case.branch(:,BR_R) = old_case.branch(:,3);%/(old_case.basekV)^2*old_case.baseMVA;% column 3
new_case.branch(:,BR_X) = old_case.branch(:,4);%/(old_case.basekV)^2*old_case.baseMVA;% column 4
new_case.branch(:,BR_B) =0; %column 5 
new_case.branch(:,RATE_A) = 9900; %column 6 %???
new_case.branch(:,RATE_B) = 0; %column 7 %???
new_case.branch(:,RATE_C) = 0; %column 8 %???
new_case.branch(:,TAP) = 1; %column 9 
new_case.branch(:,SHIFT) = 0; %column 10 
new_case.branch(:,BR_STATUS) = 1; %column 11
new_case.branch(:,ANGMIN) = -361; %column 12 
new_case.branch(:,ANGMAX) = 361; %column 13 


new_case.gen=zeros(1,21);
for n = 1: 1
    new_case.gen(n,GEN_BUS) = 1; % column 1
    new_case.gen(n,PG) = 0; %column 2 % active power
    new_case.gen(n,QG) = 0; % column 3
    new_case.gen(n,QMAX) = 100; % column 4
    new_case.gen(n,QMIN) = -100; % column 5
    new_case.gen(n,VG) = 1; % column 6
    new_case.gen(n,MBASE) = new_case.baseMVA; % column 7
    new_case.gen(n,GEN_STATUS) = 1; % column 8
    new_case.gen(n,PMAX) = 100; % column 9
    new_case.gen(n,PMIN) = 0; % column 10
end