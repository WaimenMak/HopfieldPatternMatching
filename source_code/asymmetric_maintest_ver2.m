n = 2;
x_i = Scale_X(n,:);
% k = 5.5;
graph = A\forward(Func,x_i,weight,bias,k);
% imshow(reshape(graph,[q_num,q_num]),'InitialMagnification','fit')

H = (weight + weight')/2;
Phi = phi(Func,x_i,k);
max(eig(H - A*Phi))

% X= [-0.8, 0.8, 0.8, 0.8, 0.8, -0.8, -0.8, 0.8;
%     0.8, -0.8, 0.8, -0.8, 0.8, 0.8, 0.8,-0.8;
%     0.8, 0.8, -0.8, -0.8, 0.8, 0.8, -0.8, -0.8;
%     -0.8, -0.8, 0.8, -0.8, -0.8, 0.8, 0.8, -0.8];
% 
% 
% 
% M = [0.646 -0.037 -0.072 -0.160 0.559 0.182 0.035 -0.144;
% -0.294 0.462 -0.188 -0.033 0.294 0.143 -0.554 0.070;
% -0.290 -0.357 0.665 -0.287 0.290 -0.373 0.183 0.164;
% -0.273 0.103 0.277 0.130 0.273 -0.251 -0.173 0.275;
% 0.559 0.037 0.072 0.160 0.646 -0.182 -0.035 0.144;
% 0.133 0.083 -0.103 -0.558 -0.133 0.193 0.185 -0.217;
% -0.004 -0.386 0.351 0.254 0.004 0.516 0.467 -0.094;
% 0.137 0.299 0.088 0.643 -0.137 -0.206 0.210 0.405;];
% 
% v = ones(1,8);
% A = diag(v);
% 
% H = (M + M')/2;
% Phi = phi(X(4,:));
% max(eig(H - A*Phi))

%%update
% x = [4.4,5.1,-3.9,6.2,5.7,2.4,-3.9,-4.9,-3.0];
% x = [-1.7 1.6 -2 -1.5 1.5 -1.8 1.3 -1.7 -1.9];
x = x_i;
% x = [-7.0, -6.6, 8.5, -5.6, 8.0, -6.5, 7.8,-9.3];
iter = 300;
a = [x(1)];
for t = 1:iter
%     x = (weight*activation(x)'+bias)';
    x_h = A\(weight*activation(Func,x,3,k)'+bias);
    d_x = x_h - x';

    x = x + 0.01*d_x';

    a = [a x(1)];

end
plot(1:iter+1,a);


function opt = phi(Func,x,k)
%     diag_ele = 1./derivative4(Func,x);
    diag_ele = 1./derivative3(x,k);
    
    opt = diag(diag_ele);
end

function opt = forward(Func,x,w,b,k)
    opt = w * activation(Func,x,3,k)' + b;
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



function opt = derivative(x)   %% 2/1+e^-2x -1
    opt = (4*exp(-2*x))./(1+exp(-2*x)).^2;
end

function opt = derivative2(x)
    opt = 1 - (exp(x) - exp(-x)).^2./(exp(x)+exp(-x)).^2;
end

function opt = derivative3(x,k)
%     k = 5.5;
    opt = k*(1 - (exp(k*x) - exp(-k*x)).^2./(exp(k*x)+exp(-k*x)).^2);
end

function opt = derivative4(Func,x)
    opt = 10*Func.sigmoid(x).*(1 - Func.sigmoid(x));
end

