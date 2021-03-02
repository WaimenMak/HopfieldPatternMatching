%% Pattern Template and generate synthetic data
Func = Utils;
N = 11;           %Template number
q_num = 7;
scale = 49;          %Time scaling
grid = 12;

Template.PIP = zeros(N, q_num);
% Template.represent = zeros(q_num, q_num, N);
% Template.represent = zeros(q_num*2, q_num*2, N);
Template.q_num = q_num;

%% predefined pattern
x = 1:q_num;
Template.PT = [0, 0.65, 0.3, 0.85, 0.3, 0.65, 0;     
              0.7, 0, 0.8, 0.2, 0.8, 0.4, 1;%               0, 0.7, 0.4, 0.8, 0.6, 0.9 1;
              1, 0.37, 0, 0.37, 1, 0.5, 1;
              1-1, 1-0.37, 1-0, 1-0.37, 1-1, 1-0.5, 1-1;
              1, 0, 0.7, 0, 0.7, 0, 1;      
              0, 0.7, 1, 0.5, 1, 0.7, 0;
              1-0, 1-0.7, 1-1, 1-0.5, 1-1, 1-0.7, 1-0;
              0 0.3 0.2 0.95 0.2 0.3 0;%               0, 0.65, 0.82, 0.86, 0.82, 0.65, 0;
              1-0 1-0.3 1-0.2 1-0.95 1-0.2 1-0.3 1-0;
              0 , 1, 0.6, 0.9, 0.7, 0.8, 0.75;
              0.2, 1, 0.5, 0.85,0.4,0.7,0];   %…œ…˝∆Ï–Œ
  
          
PT = Template.PT;

for i = 1:size(PT,1)
   PT(i,:) = Func.Norm(PT(i,:)); 
end

% data = Generate_data(100,scale,N,q_num, PT,0.15);  %number of warping pattern, scaling, number of template
data = Generate_negative(200,scale,N,q_num, PT,0.25);

%enlarge template
scaled_pt = zeros(N,scale);
for i =1:N
    scaled_pt(i,:) = scale_data(scale,PT(i,:),q_num);
end

for i = 1:N
%     Template.represent1(:,:,i) = Func.Grid4(PT(i,:),q_num);
    Template.represent(:,:,i) = Func.Grid8(scaled_pt(i,:),grid);
end


% plot(x,PT(2,:),'-*');
% figure(2);
% imshow(Template.represent(:,:,2),'InitialMagnification','fit')

neuron_num = size(Template.represent(:,:,1),1)^2;


plt_data(N, scale, data, q_num, Template);

X = zeros(N,neuron_num);
for i = 1:N
    p = reshape(Template.represent(:,:,i),[neuron_num, 1]);
    X(i,:) = p;
end


%% plot data
function plt_data(N, m, data, q_num, Template)
figure(1);
cc = 1;
for u = 1:N*4
    subplot(N,4,u);
%     ind = mod(u,4);
%     if ind == 0
%         ind = N;
%     end
    plot(1:m,data(u,:,cc))
    title(['pattern' num2str(cc)]);
    if mod(u,4) == 0
        cc = cc + 1;
    end
end
% % 
%% original pattern
figure(2);
for u = 1:N
    subplot(2,6,u);
    plot(1:q_num,Template.PT(u,:),'-*')
    title(['pattern' num2str(u)]);

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
