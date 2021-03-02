 %% Synthetic data
 function Syn_P = Generate_negative(number,m, N,q_num, PT,nl)   %N:number of stored patterns, number: test set size of one pattern like H&S, m:scaling size
    Syn_P = zeros(number,m, N);                             %nl: noise level;
    for n = 1:N
        for n2 = 1:number
            Syn_P(n2,:,n) = Synthetic_data(m, q_num, PT(n,:),nl);
        end
    end
    
end
 
function P_warp = Synthetic_data(m, q_num, PT, nl) %PT:predefined pattern (PIP data)

% PT = [0, 0.7, 0.3, 1, 0.3, 0.7, 0;
%       0.1, 0.3, 0.5, 0.5, 0.5,0.8, 1;
%       0, 1, 0.3, 1, 0.3, 1, 0;
%       0, 0.7, 0.4, 0.8, 0.6, 0.9 1];

    
x = 1:q_num;
% m = 49;       %91,133
n = q_num;

P_warp = Cre_data(m, n, x, PT, q_num,nl);

% plot(x_warp,P_warp,'-*')
% figure(2)
% plot(x,PT,'-*')
end

function P = Cre_data(m, n, x, PT, q_num,noise_level)
    %% Time Scaling
    X = (m - n)/(n - 1);
    for i = 2:q_num
        x(i) = x(i - 1) + (X + 1);
    end
%     plot(x,PT(1,:),'-*')

    %% Time warping
    for j = 2:q_num -1
%         x(j) =  round(x(j)-(x(j) - x(j-1))*2/3 + ((x(j)+(x(j+1) - x(j))*2/3) - (x(j)-(x(j) - x(j-1))*2/3))*rand(1));
        x(j) =  round(x(j)-(x(j) - x(j-1))*2.8/3 + ((x(j)+(x(j+1) - x(j))*2.8/3) - (x(j)-(x(j) - x(j-1))*2.8/3))*rand(1));
%         x(j) =  round(x(j-1) + (x(j+1)-x(j-1))*rand(1));

    end


    %% Time Scaling

    P = zeros(1, m);
    for c = 1:q_num - 1
        j = 0;
        for i = x(c):x(c+1)
            P(i) = PT(c) + j * (PT(c+1) - PT(c))/(x(c+1) - x(c));
            j = j+1;
        end
    end
    
    %% Noise Adding
    for u = 1:m - 1
        if rand < 0.7
%             A = 0.3*rand;
%             A = 3*rand;
            P(u) = P(u) + normrnd(0,noise_level);   %sigma = 0.05  noise level  0.1:high 0.05:low
%             P(u) = P(u) + A*(P(u+1) - P(u));
        end
    end
   
end
