%% Run after Template_DB to get data
Func = Utils;
nn = 6;         %pattern's number
set_num = 85;

Num = N;
threshold_dtw = 17;


% figsize = size(Template.represent(:,:,1),1);

t1 = clock;

g = data(set_num,:,nn);
g = Func.Norm(g);   %[-1,1]

%matching
x_query = g;

cnt = 0;
mindist = threshold_dtw;
for p = 1:N
    scale_pt = scale_data(scale, PT(p,:), q_num); %PT:[-1,1]
    dist = DTW(scale_pt,g);
    if dist < mindist
        mindist = dist;
        cnt = p;
    end
end
x_q = PT(cnt,:);

t2 = clock;
etime(t2,t1)

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
% a = [1 6 2 3 0 9 4 3 6 3];
% b = [1 3 4 9 8 2 1 5 7 3];
% w = DTW(a,b)
function w = DTW(S, Q)
    n = size(S,2);
    m = size(Q,2);
    mat = ones(n,m);

    for i = 1:n
        for j = 1:m
            mat(i,j) = abs(S(i) - Q(j));
            if (i == 1 && j >=2)
                mat(i,j) = mat(i,j) + mat(i,j-1);
            elseif (j == 1 && i >=2)
                mat(i,j) = mat(i,j) + mat(i-1,j);
            elseif (i>=2 && j>=2)
                mat(i,j) = mat(i,j) + min([mat(i-1,j-1),mat(i,j-1),mat(i-1,j)]);
            end 
        end
    end
    i = n;j = m;
    w = mat(i,j);
    
    while (i ~= 0 && j ~= 0)
        minval = 100;
        o = 0;b = 0;
        if (j == 1 && i > 1)
            w = w + mat(i-1,j);
            i = i-1;
        elseif(i == 1 && j > 1)
            w = w + mat(i,j-1);
            j = j-1;
        elseif (i == 1 && j == 1)
             w = w + mat(1,1);
             i = i-1;
        else
            if mat(i-1,j-1) < minval
                minval = mat(i-1,j-1);
                o = i-1;
                b = j-1;
            end
            if mat(i-1,j) < minval
                minval = mat(i-1,j);
                o = i-1;
                b = j;
            end
            if mat(i,j-1) < minval
                minval = mat(i,j-1);
                b = j-1;
                o = i;
            end
            w = w + minval;
            i = o;
            j = b;
        end
    end
    w = sqrt(w);
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