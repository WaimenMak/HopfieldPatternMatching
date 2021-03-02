%% asymmetric hopfield net
Func = Utils;
X = Template.PT;
scale_num = 25;
k = 5.9;
Scale_X = zeros(size(X,1),scale_num);
for i2 = 1:size(X,1)      %[-1,1]
   X(i2,:) = Func.Norm(X(i2,:)) ;
   Scale_X(i2,:) = scale_data(scale_num,X(i2,:),q_num);
   ind = abs(Scale_X(i2,:)) < 1e-10;
   Scale_X(i2,ind) = 0;
end



weight = -1 + 2*rand(size(Scale_X,2),size(Scale_X,2));  %range(-1,1)
for j = 1:size(Scale_X,2)
    weight(j,j) = -15;
end
weight(9,9) = -20;
bias = -1 + 2*rand(size(Scale_X,2),1);
v = 1.5*ones(1,size(Scale_X,2));
% v(3) = 3;
% v = [1.4,1.2,1.5,0.9,1,0.7,1];
A = diag(v);

iter = 2000;
for t = 1:iter
    for i = 1:size(Scale_X,1)
        opt = forward(Scale_X(i,:),weight,bias,Func,k);
        y = A*Scale_X(i,:)';
        [weight,bias] = backprob(weight,bias,Scale_X(i,:),opt,y,Func,k);
    end
    loss = 0;
    for item = 1:size(Scale_X,1)
        loss = loss + sum((forward(Scale_X(item,:),weight,bias,Func,k) - A*Scale_X(item,:)').^2);
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
