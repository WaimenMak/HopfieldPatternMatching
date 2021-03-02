%% Asymmetric Hopfield, run after Template_DB  Ver_2
Func = Utils;

n = N;    %the number of stored pattern
% neuron_num = 49;  %!!!
neuron_num = size(Template.represent1(:,:,1),1)^2;
%%testing
p_accuracy = [];
iter = 300;
for nn = 1:n

    correct = 0;
    for t = 1:size(data,1)
        
        r = Func.PIP(data(t,:,nn),1:size(data,2),q_num);
        g = Func.Grid4(data(t,r,nn),q_num);
        % imshow(g,'InitialMagnification','fit')

        x_query = reshape(g,[1,neuron_num]);
        x_q = x_query;

        for j = 1:iter
            x_h = A\(weight*activation(x_q,3)'+bias);
            d_x = x_h - x_q';

            x_q = x_q + 0.1*d_x';
        end

        
        cnt = 0;
        for ii = 1:n
            grap = Template.represent1(:,:,ii);
            grap = reshape(grap,[neuron_num,1]);
            if sum(abs(x_q - grap')) < 0.3          %threshold
               cnt = ii;
               break;
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

function opt = activation(x,c)
    if c == 1
        opt = 2./(1+exp(-2*x))-1;
    elseif c == 2
        opt = (exp(x) - exp(-x))./(exp(x)+exp(-x));
    elseif c == 3
        opt = (exp(6*(x)) - exp(-(6*x)))./(exp(6*(x)) + exp(-(6*x)));
    end
end