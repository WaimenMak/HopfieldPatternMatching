%% Asymmetric Hopfield, run after Template_DB  Ver_2
Func = Utils;

n = N;    %the number of stored pattern
% neuron_num = 49;  %!!!

%%testing
p_accuracy = [];
iter = 500;
thres = threshold;
min_thres = thres;
for nn = 1:n

    correct = 0;
    for t = 1:size(data,1)
        
        r = Func.PIP(data(t,:,nn),1:size(data,2),q_num);
%         g = data(t,r,nn);
        g = data(t,r,nn);
        g = scale_data(scale_num,g,q_num);
        
        
%         g = Scale_X(nn,:);
        % imshow(g,'InitialMagnification','fit')

        x_query = reshape(g,[1,neuron_num]);
        x_q = x_query;

        for j = 1:iter
            x_h = A\(weight*activation(Func,x_q,3,k)'+bias);
            d_x = x_h - x_q';

            x_q = x_q + 0.01*d_x';
        end

        
        cnt = 0;
%         for ii = 1:n
%             grap = Scale_X(ii,:);
%             grap = reshape(grap,[neuron_num,1]);
%             if sum(abs(x_q - grap')) < 0.6         %threshold
%                cnt = ii;
%                break;
%             end
%         end
        min_thres = thres;
        for ii = 1:n
            grap = Scale_X(ii,:);
            grap = reshape(grap,[neuron_num,1]);
            if sum(abs(x_q - grap')) < min_thres         %threshold
               min_thres = sum(abs(x_q - grap'));
               cnt = ii;
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

function opt = activation(Func,inpt,c,k)
if c == 1
    opt = 2./(1+exp(-2*inpt))-1;
end
if c == 2 
    opt = (exp(inpt) - exp(-inpt))./(exp(inpt)+exp(-inpt));
elseif c == 3
    opt = Func.tanh2(inpt,k);
elseif c == 4
    opt = Func.sigmoid(inpt);
end
end

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