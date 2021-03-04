%% Run after Template_DB to get data
Func = Utils;
nn = 6 ;         %pattern's number
set_num = 85;
opt = 49;
Num = N;
threshold = 3; %threshold: 0.15->0.6; 0.2-->3
neuron_num = scale_num;
t1 = clock;
r = Func.PIP(data(set_num,:,nn),1:size(data,2),q_num);



g = data(set_num,r,nn);
g = scale_data(scale_num,g,q_num);

% g = Scale_X(3,:);
% imshow(g,'InitialMagnification','fit')


% imshow(g,'InitialMagnification','fit')

x_query = reshape(g,[1, neuron_num]);
x_q = x_query;


%% Update

iter = 500;
% E = zeros(1,iter);       %Lyapunov Energy function

%tanh
% for t = 0:iter
% %     E(t+1) = -1/2*x_q'*W*x_q + x_q*b
%     j = mod(t,neuron_num);    %step by step 
% %     j = unidrnd(neuron_num - 1);  %random
%     x_q(j+1) = Func.tanh(W(j+1,:)*x_q);
% 
%     
% end
% a = [x_q(2)];
for t = 1:iter
%     E(t) = -1/2*x_q*weight*x_q' + x_q*bias;
    x_h = A\(weight*activation(Func,x_q,3)'+bias);
%     x_h = A\(weight*Lee_os(x_q,Lee,t)+bias);
    d_x = x_h - x_q';

    x_q = x_q + 0.01*d_x';

%     a = [a x_q(2)];

end

t2 = clock;
etime(t2,t1)

d = x_query;
result = x_q;

figure;
for uu = 1:size(PT,1)
    subplot(2,size(PT,1),uu);
    plot(1:scale_num,Scale_X(uu,:),'-*');
%     imshow(Template.represent(:,:,uu),'InitialMagnification','fit')
    title(['Pattern ' num2str(uu)])
end
   

subplot(2,size(PT,1),uu+1);
plot(1:scale_num,d,'-*');
% imshow(d,'InitialMagnification','fit')
title(['Query Synthetic' num2str(nn)])
subplot(2,size(PT,1),uu+2);
plot(1:scale_num,result,'-*');
% imshow(result,'InitialMagnification','fit')
title('result')

%% test module
rr = 1:scale;
subplot(2,size(PT,1),uu+3)
plot(rr(r),data(set_num,r,nn),'-*');
hold on 
plot(1:size(data,2),data(set_num,:,nn));
title('original')

cnt = 0;
for i = 1:Num
    grap = Scale_X(i,:);
%     grap = Template.Encod_represent(:,:,i);
    
    if sum(abs(x_q - grap)) < threshold              
        cnt = i;
        break;
    end
end
subplot(2,size(PT,1),uu+4)
% plot(1:t,E);
% subplot(2,size(PT,1),uu+5)
plot(1:q_num, Template.PT(cnt,:),'-*')
title(['Matching Template' num2str(cnt)])

function opt = activation(Func,inpt,c)
if c == 1
    opt = 2./(1+exp(-2*inpt))-1;
end
if c == 2 
    opt = (exp(inpt) - exp(-inpt))./(exp(inpt)+exp(-inpt));
elseif c == 3
    opt = Func.tanh2(inpt,5.5);
elseif c == 4
    opt = Func.sigmoid(inpt);
end
end
    
function output = Lee_os(x, Lee_tab,t)
N = length(x);
output = zeros(1,N);
for i = 1:N
    if x(i) < -1
        output(i) = 0.0001;
%         output(i) = -0.9999;
        output(i) = 0;
    elseif x(i) > 1
%         output(i) = 0.9999;
%         output(i) = 0.0001;
%         output(i) = 1;
        output(i) = 0.55;
    else
        rowindex = floor((x(i) - (-1))/0.002) + 1;   %¼ä¸ô0.002
%         colindex = randi([1 100]);
        colindex = t;
        output(i) = Lee_tab(rowindex, colindex);
    end
end
output = output';
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