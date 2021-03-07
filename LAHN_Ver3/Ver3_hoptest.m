%% Asymmetric Hopfield, run after Template_DB  Ver_2
Func = Utils;

n = N;    %the number of stored pattern
% neuron_num = 12*12;  %!!!
neuron_num = size(Template.represent(:,:,1),1)^2;
%%testing
p_accuracy = [];
iter = 1000;
thres = 0.1;
min_thres = thres;
for nn = 1:n

    correct = 0;
    for t = 1:size(data,1)
        
        g = Func.Grid8(data(t,:,nn),grid);
        % imshow(g,'InitialMagnification','fit')

        x_query = reshape(g,[1,neuron_num]);
        x_q = x_query;

        for j = 1:iter
            x_h = A\(weight*activation(Func,x_q,3,k)'+bias);
            d_x = x_h - x_q';

            x_q = x_q + 0.1*d_x';
        end

        
        cnt = 0;
        for ii = 1:n
            grap = Template_pt(:,:,ii);
            grap = reshape(grap,[neuron_num,1]);
            if sum(abs(x_q - grap')) < 0.1          %threshold
               cnt = ii;
               break;
            end
        end
        
%         min_thres = thres;
%         for ii = 1:n
%             grap = Template_pt(:,:,ii);
%             grap = reshape(grap,[neuron_num,1]);
%             if sum(abs(x_q - grap')) < min_thres         %threshold
%                min_thres = sum(abs(x_q - grap'));
%                cnt = ii;
%             end
%         end

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
    opt = Func.tanh2(inpt,k); %     k = 2.5
elseif c == 4
    opt = Func.sigmoid(inpt);
end
end