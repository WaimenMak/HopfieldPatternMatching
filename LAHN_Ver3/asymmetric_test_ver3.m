%% Run after Template_DB to get data
Func = Utils;
nn = 5 ;         %pattern's number
set_num = 53;
opt = 90;
Num = N;

figsize = size(Template_pt(:,:,1),1);


% g = Func.Grid4(data(set_num,r,nn),q_num);
g = Func.Grid8(data(set_num,:,nn),grid);

% g = Template_pt(:,:,6);
% imshow(g,'InitialMagnification','fit')

x_query = reshape(g,[1, neuron_num]);
x_q = x_query;


%% Update

iter = 600;
E = zeros(1,iter);       %Lyapunov Energy function

% a = [x_q(2)];
for t = 1:iter
    E(t) = -1/2*x_q*weight*x_q' + x_q*bias;
    x_h = A\(weight*activation(Func,x_q,3,k)'+bias);
%     x_h = A\(weight*Lee_os(x_q,Lee,t)+bias);
    d_x = x_h - x_q';

    x_q = x_q + 0.1*d_x';

%     a = [a x_q(2)];

end
% x_q = Func.sign(x_q);


d = reshape(x_query,[figsize,figsize]);
result = reshape(x_q,[figsize,figsize]);

figure;
for uu = 1:size(PT,1)
    subplot(2,size(PT,1),uu);
    imshow(Template_pt(:,:,uu),'InitialMagnification','fit')
    title(['Pattern' num2str(uu)])
end
   

subplot(2,size(PT,1),uu+1);
imshow(d,'InitialMagnification','fit')
title(['Query Synthetic' num2str(nn)])
subplot(2,size(PT,1),uu+2);
imshow(result,'InitialMagnification','fit')
title('result')

%% test module
subplot(2,size(PT,1),uu+3)
plot(1:scale,data(set_num,:,nn))
title('original')

cnt = 0;
for i = 1:Num
    grap = Template_pt(:,:,i);
%     grap = Template.Encod_represent(:,:,i);
    grap = reshape(grap,[1,neuron_num]);
    if sum(abs(x_q - grap)) < 0.1              %threshold
        cnt = i;
        break;
    end
end
subplot(2,size(PT,1),uu+4)
plot(1:t,E);
subplot(2,size(PT,1),uu+5)
plot(1:q_num, Template.PT(cnt,:),'-*')
title(['Matching Template' num2str(cnt)])

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
