function [N_loads,Load_node_DSS,P_l,Q_l] = Generate_loads


load loadavail20150908.mat

loc = cell2mat({Load.nameopal});
N_loads = length(loc);

P_l = zeros(N_loads,length(Load(1).kW));
Q_l = zeros(N_loads,length(Load(1).kW));
Load_node_DSS = zeros(N_loads,1);


for ii = 1:N_loads
        P_l(ii,:) = Load(ii).kW;
        Q_l(ii,:) = Load(ii).kVar;
        Load_node_DSS(ii) = loc(ii);
end
