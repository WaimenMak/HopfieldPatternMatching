%% asymmetric hopfield net
Func = Utils;
X = zeros(N,neuron_num);
k = 2.5;
Template_pt = Template.represent;
for i = 1:N
    Template_pt(:,:,i) = Template_pt(:,:,i);
    p = reshape(Template_pt(:,:,i),[neuron_num, 1]);
    X(i,:) = p;
end

% X = [1.5 -1.3 -1.8 1.5 1.3 1.8 -1.2 0.9 1.7;
%     -1.7 1.6 -2 -1.5 1.5 -1.8 1.3 -1.7 -1.9;
%     -2.0 1.5 1.4 1.9 -1.5 -1.9 1.3 -1.3 1.6;
%     1.6 1.5 -1.8 1.4 -1.6 1.9 -1.8 1.4 1.8];


% X= [-0.8, 0.8, 0.8, 0.8, 0.8, -0.8, -0.8, 0.8;
%     0.8, -0.8, 0.8, -0.8, 0.8, 0.8, 0.8,-0.8;
%     0.8, 0.8, -0.8, -0.8, 0.8, 0.8, -0.8, -0.8;
%     -0.8, -0.8, 0.8, -0.8, -0.8, 0.8, 0.8, -0.8];

% ind = X == -1;
% X(ind) = -2;
% ind2 = X == 1;
% X(ind2) = 2;

weight = -1 + 2*rand(size(X,2),size(X,2));  %range(-1,1)
for j = 1:size(X,2)
    weight(j,j) = -15;
end

bias = -1 + 2*rand(size(X,2),1);
v = 10*ones(1,size(X,2));
% v = [1.2,0.9,0.5, 1.4, 1.0, 1.5, 2.0, 1.8, 2.0];
A = diag(v);

iter = 1000;
for t = 1:iter
    for i = 1:size(X,1)
        opt = forward(X(i,:),weight,bias,Func,k);
        y = A*X(i,:)';
        [weight,bias] = backprob(weight,bias,X(i,:),opt,y,Func,k);
    end
    loss = 0;
    for item = 1:size(X,1)
        loss = loss + sum((forward(X(item,:),weight,bias,Func,k) - A*X(item,:)').^2);
    end
    fprintf('%.14f\n',loss);
end




function opt = forward(x,w,b,Func,k)
    opt = w * activation(Func,x,3,k)' + b;
end

function [w,b] = backprob(w,b,x,y_h,y,Func,k)
    lr = 0.01;
    d_w = 2*(y_h - y)*activation(Func,x,3,k);
    d_b = 2*(y_h - y);
    w = w - lr*d_w;
    b = b - lr*d_b;
end


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

