function [v] = nocontrol(T,volt,pq_fluc)
% T - number of iterations

% volt: handle for solving PF
% a,b related to cost function
%pq_fluc should be a N-by-2-T tensor
if(nargin==3)
    fluc_flag=1;
else
    fluc_flag=0;
end
    

% size of network
n = size(pq_fluc,1);



%initializing variables
v = zeros(n,T); % voltage
q = zeros(n,T);


for t=1:T
    
    % ================ Linear Model for Voltage ====================
    % ================ Non-Linear Model for Voltage =================
    if(fluc_flag==0)
        v(:,t) = volt(q(:,t));
    else
        v(:,t) = volt(q(:,t),pq_fluc(:,:,t));
    end
    %V_inj_1a(:,t) = V_1a(ind_inj_bus,t);
        
    %f(t) = f_func(q(:,t));
    %fes(t) =  norm( [proj0(v(:,t)-v_bar); proj0(v_un-v(:,t)); proj0(q(:,t)-q_bar); proj0(q_un-q(:,t))]);
% 
%     if(t<T)
%         % update for the multipliers
%         for i=1:n
%             if q_hat(i,t) + xi(i,t)/c<q_un(i)
%                 xi(i,t+1) = xi(i,t) + beta*(q_hat(i,t) - q_un(i));
%             elseif q_hat(i,t) + xi(i,t)/c > q_bar(i)
%                 xi(i,t+1) = xi(i,t) + beta*(q_hat(i,t) - q_bar(i));
%             else
%                 xi(i,t+1) = (1 - beta/c)*xi(i,t);
%             end
%         end
%         %xi_bar(:,t+1) = max(xi_bar(:,t) + beta * (q_hat(:,t) - q_bar), 0);
%         %xi_un(:,t+1)  = max(xi_un(:,t) + beta * (q_un - q_hat(:,t)),0);
%         lambda_bar(:,t+1) = max(lambda_bar(:,t) + gamma* (v(:,t) - v_bar),0);
%         lambda_un(:,t+1) = max(lambda_un(:,t) + gamma*(v_un - v(:,t)),0);
    end
end