function [R,X] = generateRX(branch)
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
% branch = [
% 	1	2	0.01938	0.05917	0.0528	0	0	0	0	0	1	-360	360;
% 	1	5	0.05403	0.22304	0.0492	0	0	0	0	0	1	-360	360;
% 	2	3	0.04699	0.19797	0.0438	0	0	0	0	0	1	-360	360;
% 	2	4	0.05811	0.17632	0.034	0	0	0	0	0	1	-360	360;
% 	2	5	0.05695	0.17388	0.0346	0	0	0	0	0	1	-360	360;
% 	3	4	0.06701	0.17103	0.0128	0	0	0	0	0	1	-360	360;
% 	4	5	0.01335	0.04211	0	0	0	0	0	0	1	-360	360;
% 	4	7	0	0.20912	0	0	0	0	0.978	0	1	-360	360;
% 	4	9	0	0.55618	0	0	0	0	0.969	0	1	-360	360;
% 	5	6	0	0.25202	0	0	0	0	0.932	0	1	-360	360;
% 	6	11	0.09498	0.1989	0	0	0	0	0	0	1	-360	360;
% 	6	12	0.12291	0.25581	0	0	0	0	0	0	1	-360	360;
% 	6	13	0.06615	0.13027	0	0	0	0	0	0	1	-360	360;
% 	7	8	0	0.17615	0	0	0	0	0	0	1	-360	360;
% 	7	9	0	0.11001	0	0	0	0	0	0	1	-360	360;
% 	9	10	0.03181	0.0845	0	0	0	0	0	0	1	-360	360;
% 	9	14	0.12711	0.27038	0	0	0	0	0	0	1	-360	360;
% 	10	11	0.08205	0.19207	0	0	0	0	0	0	1	-360	360;
% 	12	13	0.22092	0.19988	0	0	0	0	0	0	1	-360	360;
% 	13	14	0.17093	0.34802	0	0	0	0	0	0	1	-360	360;
% ];
n1 = max(branch(:,1));
n2 = max(branch(:,2));
n = max(n1,n2); % number of nodes.

adj = zeros(n,n);%adjacency matrix
X_0 = zeros(n,n);%adjacency matrix with reactance as weights
R_0 = zeros(n,n);%adjacency matrix with resistance as weights
M = size(branch,1); % number of lines
for m = 1:M
    adj(branch(m,1),branch(m,2)) = 1;
    adj(branch(m,2),branch(m,1)) = 1;
    
    R_0(branch(m,1),branch(m,2)) = branch(m,3);
    R_0(branch(m,2),branch(m,1)) = branch(m,3);
    
    X_0(branch(m,1),branch(m,2)) = branch(m,4);
    X_0(branch(m,2),branch(m,1)) = branch(m,4);
end

X = zeros(n-1,n-1);
R = zeros(n-1,n-1);
for k1 = 2:n
    for k2 = 2:n
        path1 = find_path_to_root(branch(:,1:2),k1);
        path2 = find_path_to_root(branch(:,1:2),k2);
        path = intersect(path1,path2);
        r_tmp = 0;
        x_tmp = 0;
        for kk = 1:length(path)-1
            x_tmp = x_tmp +  2*X_0(path(kk),path(kk+1));
            r_tmp = r_tmp + 2* R_0(path(kk),path(kk+1));
        end
        X(k1-1,k2-1) = x_tmp;
        R(k1-1,k2-1) = r_tmp;
    end
end
function path = find_path_to_root (paths, node)
    % find the path from 1 to node
    path = node;
    current = node; 
    while 1
        next_idx = find(paths(:,2) == current);
        next = paths(next_idx,1);
        path = [next,path];
        if(next == 1)
            break;
        end
        current = next;
    end


