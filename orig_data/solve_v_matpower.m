function v = solve_v_matpower(case_mpc,q, loading)
define_constants;
%n = size(q,1);
%v = zeros(n,1);
%case_mpc.bus(2:end,PD) =-p;
%loading(:,1) should provide the vector of real power, loading(:,2)
%reactive power, in the form of injection
if(nargin==3)
    case_mpc.bus(2:end,PD) = -loading(:,1);
    case_mpc.bus(2:end,QD) = -loading(:,2);
end
case_mpc.bus(2:end,QD) =case_mpc.bus(2:end,QD) -q;
pfresult = runpf(case_mpc);
v = (pfresult.bus(2:end,VM)).^2;
