%%ED matching
Func = Utils;
nn = 6;         %pattern's number
set_num = 85;

Num = N;
threshold_ED = 3;


% figsize = size(Template.represent(:,:,1),1);


g = data(set_num,:,nn);
g = Func.Norm(g);   %[-1,1]

%matching
x_query = g;

cnt = 0;
mindist = threshold_ED;
for p = 1:N
    scale_pt = scale_data(scale, PT(p,:), q_num); %PT:[-1,1]
    dist = ED_Dist(scale_pt,g);
    if dist < mindist
        mindist = dist;
        cnt = p;
    end
end
x_q = PT(cnt,:);
d = x_query;
result = scale_data(scale, x_q, q_num);
r = Func.PIP(result,1:size(data,2),q_num);

figure;
for uu = 1:size(PT,1)
    subplot(2,size(PT,1),uu);
    plot(1:q_num,PT(uu,:),'-*');
%     imshow(Template.represent(:,:,uu),'InitialMagnification','fit')
    title(['Pattern' num2str(uu)])
end
   

subplot(2,size(PT,1),uu+1);
plot(1:scale,d,'-*');
% imshow(d,'InitialMagnification','fit')
title(['Query Synthetic' num2str(nn)])
subplot(2,size(PT,1),uu+2);
plot(1:scale,result,'-*');
% imshow(result,'InitialMagnification','fit')
title('result')

%% test module
rr = 1:scale;
subplot(2,size(PT,1),uu+3)
plot(1:scale,data(set_num,:,nn));
hold on 
plot(1:scale,result);
title('original')


subplot(2,size(PT,1),uu+4)
plot(1:q_num, Template.PT(cnt,:),'-*')
title(['Matching Template' num2str(cnt)])

    
function D = ED_Dist(SP,Q)
    n = size(SP,2);
    sum = 0;
    for i = 1:n
        sum = sum + (SP(i) - Q(i))^2;
    end
    D = sqrt(sum);
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