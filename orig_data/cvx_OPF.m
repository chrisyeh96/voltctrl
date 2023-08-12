function [opt_val,opt_q_inj,opt_v] = cvx_OPF(G,v_min,v_max,q_min,q_max,X,a,b,d)
% G - problem data
% cost: 1/2*a*q^2 + b*q + 1/2*d*q^T X q (need parameter X,a,b,d)

n = size(v_min,1); % number of load buses, excluding source bus
r = G.branch(:,3);
x = G.branch(:,4);
p_non = G.bus(2:end,2); % non-controllable p load
q_non = G.bus(2:end,3); % non-controllable q load
cvx_begin
%cvx_quiet(true)
cvx_precision('best')
    variables P(n) Q(n) V(n+1) I(n) q_inj(n);
    minimize sum(1/2*a.*(q_inj.^2)) + sum(b.*q_inj) + 1/2*d*transpose(q_inj)*X*q_inj 
    %+ 0.1*sum(I.^2)
    %  (r'*I+coief*(p_DR-pmax).^2+coe0a*square(P(1))+coe0b*P(1));
    subject to
        V(1) == 1;
        V(2:end)>= v_min;
        V(2:end)<= v_max;
        q_inj <= q_max;
        q_inj >= q_min; %pmax*0.5;
        for i=1:n  % go through all branches, branch #i connects node i+1 to its parent. Node (i+1) corresponds to q_inj index of i
            V(G.branch(i,2)) == V(G.branch(i,1)) - ...
                2*(r(i)*P(i)+x(i)*Q(i)) + (r(i)^2+x(i)^2)*I(i);
            P(i) == p_non(i) + r(i)*I(i) + ... 
                sum(P(find(G.branch(:,1)==G.branch(i,2))));
            Q(i) == -q_inj(i)+ q_non(i) + x(i)*I(i) + ...
                sum(Q(find(G.branch(:,1)==G.branch(i,2))));
            I(i) >= quad_over_lin([P(i);Q(i)],V(G.branch(i,1)));
        end
cvx_end

opt_val = cvx_optval;
opt_q_inj = q_inj;
opt_v = V;
% tightness check
        for i=1:n  % go through all branches, branch #i connects node i+1 to its parent. Node (i+1) corresponds to q_inj index of i
            if(I(i) > quad_over_lin([P(i);Q(i)],V(G.branch(i,1)))+1e-6)
                display('OPF solver not tight');
            end
        end

