%% Asymmetric Hopfield, run after Template_DB  Ver_2
Func = Utils;
% scale_num = 49;
n = N;    %the number of stored pattern
% neuron_num = 49;  %!!!

%%testing
p_accuracy = [];

for nn = 1:n

    correct = 0;
    
    for t = 1:size(data,1)
        r = Func.PIP(data(t,:,nn),1:size(data,2),q_num);

        g = data(t,r,nn);
        g = Func.Norm(g);   %[-1,1]
        
        cnt = 0;
        mindist = threshold_pip;
        for p = 1:N
            scale_pt = scale_data(scale, PT(p,:), q_num); %PT:[-1,1]
            r2 = Func.PIP(scale_pt,1:size(data,2),q_num);
            dist = Dist(scale_pt(r2),g,r,r2);
%             dist = ED_Dist(scale_pt(r2),g);
            if dist < mindist
                mindist = dist;
                cnt = p;
            end
        end

        if cnt == nn
            correct = correct+1;
        end
    end
    p_accuracy = [p_accuracy correct];
end


bar(p_accuracy)
title('Accuracy');

function P = scale_data(m, PT, q_num)
    n = q_num;
    x = 1:q_num;
    %% Time Scaling
    X = (m - n)/(n - 1);
    for i = 2:q_num
     x(i) = x(i - 1) + (X + 1);
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
end

function D = ED_Dist(SP,Q)
    n = size(SP,2);
    sum = 0;
    for i = 1:n
        sum = sum + (SP(i) - Q(i))^2;
    end
    D = sqrt(sum);
end

function D = Dist(SP,Q,ind,ind_pt)
    w = 0.5;
    n = size(SP,2);
    q_num = size(SP,2);
    sum_h = 0;    %horizontal
    sum_v = 0;    %vertical
    for i = 1:q_num
        sum_h = sum_h + (SP(i) - Q(i))^2;
        if i > 1
            sum_v = sum_v + (ind_pt(i) - ind(i))^2;
        end
    end
    AD = sqrt(1/n*sum_h);
    TD = sqrt(1/(n-1)*sum_v);
    D = w*AD+(1-w)*TD;
end