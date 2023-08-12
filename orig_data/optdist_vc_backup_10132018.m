function [v, q,fes, f] = optdist_vc(T,stepsize,v_un,v_bar,q_un,q_bar,volt,a,b,Y,pq_fluc)
% T - number of iterations
% stepsize - 3-vecotr containing value of alpha,beta,gamma
% v_un,v_bar: n-by-1 vector of voltage lower/upper bound
% q_un, q_bar: n-by-1 vector of quadratic power lower/upper bound
% volt: handle for solving PF
% a,b related to cost function
% Y: inv of X
%pq_fluc should be a N-by-2-T tensor
if(nargin==11)
    fluc_flag=1;
else
    fluc_flag=0;
end
    

% size of network
n = size(q_un,1);

% step sizes
alpha = stepsize(1);
beta = stepsize(2);
gamma = stepsize(3);

%initializing variables
q_hat = zeros(n,T); % ''virtual'' reactive power
xi = zeros(n,T); % lagrangian multiplier for reactive power constraint
lambda_bar = zeros(n,T); % lagrangian multipler for voltage constraint (upper limit)
lambda_un = zeros(n,T); % lagrangian multipler for voltage constraint (lower limit)
v = zeros(n,T); % voltage
q = zeros(n,T); % ''actual'' reactive power
f = zeros(1,T); % objective function value
fes = zeros(1,T); % feasibility of solution
c=1;% parameter for augmented Lagrangian

% projection functions
proj0 = @(r) max(r,zeros(size(r)));


% gradient handle
nabla = @(qqq) a.*qqq + b;
f_func = @(qqq) sum(1/2*a.*(qqq.^2) + b.*qqq);

for t=1:T
    if(t>1)
        GG = max(min(xi(:,t) + c*(q_hat(:,t-1) - q_un),0), xi(:,t)+c*(q_hat(:,t-1) - q_bar));
        q_hat(:,t) = q_hat(:,t-1) - alpha*(lambda_bar(:,t) - lambda_un(:,t) + Y*(nabla(q_hat(:,t-1)) + GG ));
    end
    %q_hat(:,t) = X*(lambda_un(:,t) - lambda_bar(:,t)) + xi_un(:,t) - xi_bar(:,t);
    q(:,t) = max(min(q_hat(:,t),q_bar),q_un); % ''actuall implemented reactive power''
    
    % ================ Linear Model for Voltage ====================
    % ================ Non-Linear Model for Voltage =================
    if(fluc_flag==0)
        v(:,t) = volt(q(:,t));
    else
        v(:,t) = volt(q(:,t),pq_fluc(:,:,t));
    end
    %V_inj_1a(:,t) = V_1a(ind_inj_bus,t);
        
    f(t) = f_func(q(:,t));
    fes(t) =  norm( [proj0(v(:,t)-v_bar); proj0(v_un-v(:,t)); proj0(q(:,t)-q_bar); proj0(q_un-q(:,t))]);

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
%transpose
%V=v;
%Q=q;
%fes=fes;
%f=f;

