%% Processing time test
Func = Utils;
nn = 1;         %pattern's number
set_num = 1;

Num = N;
threshold_dtw = 100;
threshold_ED = 100;
threshold_pip = 100;

% figsize = size(Template.represent(:,:,1),1);

T1 = [];
T2 = [];
T3 = [];
T4 = [];
scale_len = zeros(1,10);
scale_len(1) = 49;
for j = 2:10
    scale_len(j) = scale_len(j-1)+42*3;
end

%% ED
for i = 1:10
    data = Generate_negative(1,scale_len(i),N,q_num, PT,0.2);
    g = data(set_num,:,nn);
    g = Func.Norm(g);   %[-1,1]
    t1 = clock;
    x_query = g;

    cnt = 0;
    mindist = threshold_ED;
    for p = 1:N
        scale_pt = scale_data(scale_len(i), PT(p,:), q_num); %PT:[-1,1]
        dist = ED_Dist(scale_pt,g);
        if dist < mindist
            mindist = dist;
            cnt = p;
        end
    end
    x_q = PT(cnt,:);
    t2 = clock;
    T1 = [T1 etime(t2,t1)];

%% PIP
    t1 = clock;
    r = Func.PIP(data(set_num,:,nn),1:size(data,2),q_num);
    x_query = g;
    cnt = 0;
    mindist = threshold_pip;
    for p = 1:N
        scale_pt = scale_data(scale_len(i), PT(p,:), q_num); %PT:[-1,1]
        r2 = Func.PIP(scale_pt,1:size(data,2),q_num);
        dist = Dist(scale_pt(r2),g,r,r2);
        if dist < mindist
            mindist = dist;
            cnt = p;
        end
    end
%     x_q = PT(cnt,:);
    t2 = clock;
    T2 = [T2 etime(t2,t1)];


%% scaling PIP
    t1 = clock;
    
    %scaling
%     r = Func.PIP(data(set_num,:,nn),1:size(data,2),q_num);
%     pic = data(set_num,r,nn);
%     g_scale = scale_data(scale_num,pic,q_num); 
%     x_query = reshape(g_scale,[1, neuron_num]);
%     x_q = x_query;
%     iter = 500;
%     for t = 1:iter
%         x_h = A\(weight*activation(Func,x_q,3,5.5)'+bias);
% 
%         d_x = x_h - x_q';
% 
%         x_q = x_q + 0.01*d_x';
%     end
    
    %n-equal-part
%     grid = Func.Grid8(data(set_num,:,nn),12);
%     x_query = reshape(grid,[1, 12*12]);
%     x_q = x_query;
%     iter = 600;
%     for t = 1:iter
%         x_h = A\(weight*activation(Func,x_q,3,2.5)'+bias);
%         d_x = x_h - x_q';
%         x_q = x_q + 0.1*d_x';
%     end
%     
    %pip tg
    r_pip = Func.PIP(data(set_num,:,nn),1:size(data,2),q_num);
    graph = Func.Grid4(data(set_num,r_pip,nn),q_num);

    x_query = reshape(graph,[1, 49]);
    x_q = x_query;
    iter = 300;
    for t = 1:iter
        x_h = A\(weight*activation(Func,x_q,3,k)'+bias);
        d_x = x_h - x_q';
        x_q = x_q + 0.1*d_x';

    end    
    
    t2 = clock;
    T3 = [T3 etime(t2,t1)];

%% DTW matching
    t1 = clock;
    x_query = g;

    cnt = 0;
    mindist = threshold_dtw;
    for p = 1:N
        scale_pt = scale_data(scale_len(i), PT(p,:), q_num); %PT:[-1,1]
        dist = DTW(scale_pt,g);
        if dist < mindist
            mindist = dist;
            cnt = p;
        end
    end
    x_q = PT(cnt,:);

    t2 = clock;
    T4 = [T4 etime(t2,t1)];
end

T = [T1;T2;T3;T4];
figure;
x = scale_len;
plot(x,T1,x,T2,x,T3,x,T4);
legend();

%% template
% name = ["H&S" "Tria-A" 'CWH'	'Reverse CWH' 'Trip-B' 'Doub-T'	'Doub-B' 'Spike-T' 'Spike-B' 'Flag'	'Wedges']
% figure;
% for uu = 1:size(PT,1)
%     subplot(3,4,uu);
%     imshow(Template.represent1(:,:,uu),'InitialMagnification','fit')
%     title(name(uu))
% end



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



function D = Dist(SP,Q,ind,ind_pt)
    w = 0.5;
    n = size(SP,2);
    q_num = size(SP,2);
    sum_h = 0;    %horizontal
    sum_v = 0;    %vertical
    for i = 1:q_num
        sum_h = sum_h + (SP(i) - Q(i))^2;
        if i > 1
            sum_v = sum_v + (ind_pt(i) - ind(i))^2;
        end
    end
    AD = sqrt(1/n*sum_h);
    TD = sqrt(1/(n-1)*sum_v);
    D = w*AD+(1-w)*TD;
end

function D = ED_Dist(SP,Q)
    n = size(SP,2);
    sum = 0;
    for i = 1:n
        sum = sum + (SP(i) - Q(i))^2;
    end
    D = sqrt(sum);
end

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