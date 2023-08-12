function P_pv = Generate_PV_power(Capacity)


% Load one day of data for irradiance 
load pvavail20150908_2.mat; 

% create normalized profile
PV_profile = PVavail(1).PVp_6s./PVavail(1).PVacrate;



% Populate PV data                                       
P_pv = zeros(length(Capacity),length(PV_profile));
for ii = 1:length(Capacity);
    P_pv(ii,:) = PV_profile.*Capacity(ii);
end



