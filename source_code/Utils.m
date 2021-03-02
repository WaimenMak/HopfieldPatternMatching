function Func = Utils
    Func.sigmoid = @sigmoid;
    Func.tanh = @tanh;
    Func.tanh2 = @tanh2;
    Func.sigmoid2 = @sigmoid2;
    Func.sign2 = @sign2;
    Func.sign = @sign;
    Func.PIP = @PIP;
    Func.Norm = @Normalize;
    Func.Lee = @Lee_os;
    Func.Template = @Grid_template;
    Func.Grid = @Grid;
    Func.Grid2 = @Grid2;
    Func.Grid3 = @Grid3;
    Func.Grid4 = @Grid4;
    Func.Grid5 = @Grid5;
    Func.Grid6 = @Grid6;
    Func.Grid7 = @Grid7;
    Func.Grid8 = @Grid8;
    Func.Encoder = @Encoder;
    Func.Sliding = @Sliding_Win;
    Func.plt = @plt;
    Func.sim = @similarity;
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
        colindex = randi([1 100]);
%         colindex = t;
        output(i) = Lee_tab(rowindex, colindex);
    end
end
output = output';
end


function y = sigmoid(x)
    k = 10;
    T = 0.5;
    y = 1./(ones(size(x)) + exp(-k*(x - T)));
end

function y = sign2(x)
%     x(find(x >= 1)) = 1;
%     x(find(x < -1)) = 0;  
%     y = sign(x);

    x(find(x >= 0.5)) = 1;
    x(find(x < 0.5)) = 0;  
    y = x;

end

function y = sign(x)
%     x(find(x >= 1)) = 1;
%     x(find(x < -1)) = 0;  
%     y = sign(x);

    x(find(x > 0)) = 1;
    x(find(x <= 0)) = -1;  
    y = x;

end

function y = tanh(x)
    T = 0;       %threshold
    if x < -1
        y = 0;
    elseif x > 1
        y = 0.55;
    else
        y = (exp(6*(x - T)) - exp(-(6*x - T)))./(exp(6*(x - T)) + exp(-(6*x - T)));
    end
end

function y = tanh2(x,k)
    T = 0;       %threshold
%     k = 6;
    y = (exp(k*(x - T)) - exp(-(k*x - T)))./(exp(k*(x - T)) + exp(-(k*x - T)));
end


function y = sigmoid2(x)   %[-1,1]
    k = 6;
    if x < -1
        y = 0;
    elseif x >= 1
        y = 1;
    else
        y = (ones(size(x)) - exp(-k*x))./(ones(size(x)) + exp(-k*x));
    end
end

function pattern = Sliding_Win(data, step_size, width, num)  %num:the index of kth window (start from 0)/ /stepsize: sliding width/ /width: window width/
    pattern = data(1+num*step_size:width + num*step_size);
end


function SP = PIP(P,x,SP_num) %% perceptual important point
%P the query, Q the template,P and Q have to be normalized
SP_old = [x(1) x(end)];
SP = SP_old;

while length(SP_old) < SP_num
    max_VD = 0;
    for j = 1:length(SP_old)-1
        if length(SP) >= SP_num
            break;
        end
        for i = SP_old(j)+1:SP_old(j+1)-1
            VD = abs(P(SP_old(j)) + (P(SP_old(j+1)) - P(SP_old(j)))*(x(i)-x(SP_old(j)))/(x(SP_old(j+1))-x(SP_old(j)))-P(i));  %%VD distance
            if VD >= max_VD
                max_VD = VD;
                count = i;
            end
        end  
    end
    SP = [SP x(count)];
    SP_old = sort(SP);

end

SP = SP_old;
end


function Y = Normalize(pattern)

Y = (pattern - min(min(pattern)))*2./(max(max(pattern)) - min(min(pattern))) + (-1); %range[-1,1]

end

function output = Grid_template(x, M)   %% grid dimension (isometry)
interval = floor(length(x)/M) ;
point = zeros(1,M);
point(1) = x(1);
for i = 1:M-1
    point(i+1) = x(1+i*interval);
end
output = point;
end

function graph =Grid(ts,M)     %%grid representation
graph = -1*ones(M,M);
H = max(ts);
L = min(ts);
for i = 1:M
    pos = floor(M*(ts(i) - L)/(H - L))+1;
    if pos > M
        pos = M;
    end
    graph(M - pos + 1,i) = 1;
end
end

function graph =Grid2(ts,M)     %%grid representation version2
graph = -1*ones(M,M);
H = max(ts);
L = min(ts);

pos = floor(M*(ts(1) - L)/(H - L))+1;
if pos > M
    pos = M;
end
current = M - pos + 1;
graph(current,1) = 1;
for i = 2:M
    last_pos = current;
    pos = floor(M*(ts(i) - L)/(H - L))+1;
    if pos > M
        pos = M;
    end
    current = M - pos + 1;
    graph(current,i) = 1;
    switch (current)  %whether on the boundary
        case M
            if (graph(current,i-1) ~= 1 && graph(current - 1,i-1) ~= 1)  %the white block is not continuous
               sym = (last_pos - current)/abs(last_pos - current);
               for m = 1*sym:sym:last_pos - current + -1*sym
                   graph(current + m,i) = 1;
               end
               current = current + m;
            end

        case 1
            if (graph(current,i-1) ~= 1 && graph(current + 1,i-1) ~= 1)  %the white block is not continuous
               sym = (last_pos - current)/abs(last_pos - current);
               for m = 1*sym:sym:last_pos - current + -1*sym
                   graph(current + m,i) = 1;
               end
               current = current + m;
            end

        otherwise
            if (graph(current,i-1) ~= 1 && graph(current - 1,i-1) ~= 1 && graph(current + 1,i-1) ~= 1)  %the white block is not continuous
               sym = (last_pos - current)/abs(last_pos - current);
               for m = 1*sym:sym:last_pos - current + -1*sym
                   graph(current + m,i) = 1;
               end
               current = current + m;
            end

    end
end

end

function graph =Grid3(ts,M)     %%grid representation version2
graph = -1*ones(M,M);
% graph = zeros(M,M);
H = max(ts);
L = min(ts);

pos = floor(M*(ts(1) - L)/(H - L))+1;
if pos > M
    pos = M;
end
current = M - pos + 1;
graph(current,1) = 1;
for i = 2:M
    last_pos = current;
    pos = floor(M*(ts(i) - L)/(H - L))+1;
    if pos > M
        pos = M;
    end
    current = M - pos + 1;
    graph(current,i) = 1;
    switch (current)  %whether on the boundary
        case M
            if (graph(current,i-1) ~= 1 && graph(current - 1,i-1) ~= 1)  %the white block is not continuous
               sym = (last_pos - current)/abs(last_pos - current);
               for m = 1*sym:sym:last_pos - current + -1*sym
                   graph(last_pos - m,i) = 1;
               end

            end

        case 1
            if (graph(current,i-1) ~= 1 && graph(current + 1,i-1) ~= 1)  %the white block is not continuous
               sym = (last_pos - current)/abs(last_pos - current);
               for m = 1*sym:sym:last_pos - current + -1*sym
                   graph(last_pos - m,i) = 1;
               end

            end

        otherwise
            if (graph(current,i-1) ~= 1 && graph(current - 1,i-1) ~= 1 && graph(current + 1,i-1) ~= 1)  %the white block is not continuous
               sym = (last_pos - current)/abs(last_pos - current);
               for m = 1*sym:sym:last_pos - current + -1*sym
                   graph(last_pos - m,i) = 1;
               end

            end

    end
end

end

function graph =Grid4(ts,M)     %%grid representation
graph = -1*ones(M,M);
H = max(ts);
L = min(ts);
for i = 1:M
    pos = floor(M*(ts(i) - L)/(H - L))+1;
    if pos > M
        pos  = M;
    end
    D = 2*M/((M - pos)*(M - pos + 1)+(pos - 1)*pos);
    for j = 1:M
        graph(M - j + 1,i) = 1 - abs(pos - j)*D;
    end
end
% pos =[1 3 4 7 8 6 8 10 6 3];
% for i = 1:M
%     for j = 1:M
%         D = 2*M/((M - pos(i))*(M - pos(i) + 1)+(pos(i) - 1)*pos(i));
%         graph(M - j + 1,i) = 1 - abs(pos(i) - j)*D;
%     end
% end

end

function graph =Grid5(ts,M)     %%grid representation, similarity
graph = -1*ones(M,M);
H = max(ts);
L = min(ts);
for i = 1:M
    pos = floor(M*(ts(i) - L)/(H - L))+1;
    if pos > M
        pos = M;
    end
    D = 2*M/((M - pos)*(M - pos + 1)+(pos - 1)*pos);
    for j = 1:M
        if 1 - abs(pos - j)*D > 0
            graph(M - j + 1,i) = 1 ;
        else
            graph(M - j + 1,i) = -1 ;
        end
    end
end
% pos =[1 3 4 7 8 6 8 10 6 3];
% for i = 1:M
%     for j = 1:M
%         D = 2*M/((M - pos(i))*(M - pos(i) + 1)+(pos(i) - 1)*pos(i));
%         graph(M - j + 1,i) = 1 - abs(pos(i) - j)*D;
%     end
% end

end

function graph =Grid6(ts,M)     %%enlarge grid representation version3
time_series = zeros(1,size(ts,2)*2);
count = 1;
for k = 1:M
   time_series(count) = ts(k);
   time_series(count+1) = ts(k);
   count = count+2;
end
ts = time_series;
M = M*2;
graph = -1*ones(M,M);
% graph = zeros(M,M);
H = max(ts);
L = min(ts);

pos = floor(M*(ts(1) - L)/(H - L))+1;
if pos > M
    pos = M;
end
current = M - pos + 1;
graph(current,1) = 1;
for i = 2:M
    last_pos = current;
    pos = floor(M*(ts(i) - L)/(H - L))+1;
    if pos > M
        pos = M;
    end
    current = M - pos + 1;
    graph(current,i) = 1;
    switch (current)  %whether on the boundary
        case M
            if (graph(current,i-1) ~= 1 && graph(current - 1,i-1) ~= 1)  %the white block is not continuous
               sym = (last_pos - current)/abs(last_pos - current);
               for m = 1*sym:sym:last_pos - current + -1*sym
                   graph(last_pos - m,i) = 1;
               end

            end

        case 1
            if (graph(current,i-1) ~= 1 && graph(current + 1,i-1) ~= 1)  %the white block is not continuous
               sym = (last_pos - current)/abs(last_pos - current);
               for m = 1*sym:sym:last_pos - current + -1*sym
                   graph(last_pos - m,i) = 1;
               end

            end

        otherwise
            if (graph(current,i-1) ~= 1 && graph(current - 1,i-1) ~= 1 && graph(current + 1,i-1) ~= 1)  %the white block is not continuous
               sym = (last_pos - current)/abs(last_pos - current);
               for m = 1*sym:sym:last_pos - current + -1*sym
                   graph(last_pos - m,i) = 1;
               end

            end

    end
end

end

function graph =Grid7(temp,M)     %%enlarge grid representation version3
M = M*2;
H = max(temp);
L = min(temp);
ts = zeros(1,M);
k = 1;
for m = 1:M/2
    ts(k) = temp(m);
    ts(k+1) = temp(m);
    k = k+2;
end

for i = 1:M
    pos = floor(M*(ts(i) - L)/(H - L))+1;
    if pos > M
        pos = M;
    end
    D = 2*M/((M - pos)*(M - pos + 1)+(pos - 1)*pos);
    for j = 1:M
        graph(M - j + 1,i) = 1 - abs(pos - j)*D;
    end
end

end


function graph =Grid8(data,M)     %%enlarge grid representation version3
ts = zeros(1,M);
ts(1) = data(1);
ts(end) = data(end);
interval = floor((size(data,2) - 2)/(M - 2 + 1));
for n = 1:M-2
    ts(n+1) = data(1 + n*interval);
end
graph = -1*ones(M,M);
H = max(ts);
L = min(ts);
for i = 1:M
    pos = floor(M*(ts(i) - L)/(H - L))+1;
    if pos > M
        pos = M;
    end
    D = 2*M/((M - pos)*(M - pos + 1)+(pos - 1)*pos);
    for j = 1:M
        graph(M - j + 1,i) = 1 - abs(pos - j)*D;
    end
end

end


function code = Encoder(ts,M)
col = ceil(log2(M));
code = -1*ones(M,col);
H = max(ts);
L = min(ts);
for i = 1:M
    pos = floor(M*(ts(i) - L)/(H - L))+1;
    if pos > M
        pos = M;
    end
    col_index = pos - 1;  %% range:0~7 if M = 8
    str = dec2bin(col_index);
    len = length(str);
    for j = 1:len
        if str2double(str(len - j + 1)) == 0
            code(i,col - j + 1) = -1;
        else
            code(i,col - j + 1) = str2double(str(len - j + 1));
        end
    end
end
end

function plt(x,pattern,q_num)
    Func = Fun;
    pattern = Func.Norm(pattern);
    point = Func.PIP(pattern,x,q_num);
    g = Func.Grid2(pattern(point),q_num);
    imshow(g,'InitialMagnification','fit')
    figure(2)
    plot(x(point),pattern(point),'*-');
end

function sim = similarity(x ,y,choice) %x:x_query, y:hnn output
if choice == 1
    %cos
    sim = x'*y/size(x,1);
elseif choice == 2
%gram mattrix
    cov = x*y';
    cov = reshape(cov,[size(x,1)^2,1]);
    res = reshape(x*x',[size(x,1)^2,1]);
    sim = cov'*res/(size(x,1)^2);
else
    cov = x*y';
    res = x*x';
%     sim = 1 - abs(sum(sum(res - cov)))/(size(x,1)^2);
%     sim = 1 - sum(sum(abs(res - cov)))/(size(x,1)^2);
    sim = 1 - (sum(sum(res - cov)))/(size(x,1)^2);
end
end
