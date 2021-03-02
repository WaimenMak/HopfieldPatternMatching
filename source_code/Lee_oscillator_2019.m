% Chaotic Neuro Ocillator - Lee-Oscillator
% 
% This script will implement ROscillator:
%     u(t+1) = f(a1*u(t) - a2*v(t) + I(t) - tu)
%     v(t+1) = f(b1*u(t) - b2*v(t) - tv)
%     w(t+1) = f(I(t))
%     z(t)   =  [u(t) - v(t)]*exp(-kI(t)) + c*w(t)
%
%     where
%       tu = tv = 0
%       a1 >= 2b1, a2 >=2*b2 
%       Put a1=a2=5, b1=b2=1
%       set I(t) - the external stimulus, as a fixed value in each case
%       f(p,s) = tanh(p, s) where s is the sigma value of the tanh function
%       In this test, we try oscillator (s/a/b) = (5/5/1) with z(1) = 0.2 and -0.2
%
%   and plot the bif. function of z w.r.t Input I lies between -1 and +1
%
% Created
%   by 
%
% Raymond Lee
%
%   April 2002
% Modified on
%   19 May 2019
%%%%%%%%%%%

%%%%%%%%%%
% Define parameters
%
  N = 600;    % n = no.of time step default is 1000
  s = 6;      % parameter for tanh function
  a1 = 5;     % default is 5
  a2 = 5;
  b1 = 1;     % default is 5
  b2 = 1;
  eu = 0;     % u threshold default is 0
  ev = 0;     % v threshold defalut is 0
  k  = 500;     % Decay constant
  c  = 1;
  e  = 0.02;
  x  = 1;     % x index of Lee()
 
%%%%%%%%%%
% Define and initialize u(t) and v(t)
%
  u = zeros(1,N);
  v = zeros(1,N);
  w = zeros(1,N);
  z = zeros(1,N);
  Lee = zeros(1001,300);
  z(1) = 0.2;   % IC of z set to 0.2
  u(1) = 0.2;
  x_aix = zeros(1,1001*300);
  j = 1;  
%   Lee_2 = zeros(1,1001);
  for i=-1:0.002:1
    valueOf_stimulus_i = sprintf('%0.5g',i);
    sim = i + 0.02*sign(i);   %%向左向右移，消除不对称性  stimulus
%     sim = i;
    for t = 1:N-1     
        tempu = a1*u(t) - a2*v(t) + sim - eu;
        tempv = b1*u(t) - b2*v(t) - ev;
        u(t+1) = (exp(s*tempu) - exp((-1)*s*tempu))/(exp(s*tempu) + exp((-1)*s*tempu));
        v(t+1) = (exp(s*tempv) - exp((-1)*s*tempv))/(exp(s*tempv) + exp((-1)*s*tempv));
        w(t+1) = (exp(s*sim)   - exp((-1)*s*sim)) /(exp(s*sim) + exp((-1)*s*sim)); 
        z(t+1) = ((u(t+1) - v(t+1)) * exp (-1*k*sim*sim) + c*w(t+1)) ;       
        if (t >=300)
          x_aix(j) = i;
          j = j+1;
% 	      plot(x/1000, z(t+1)/2 + 0.5);
%           Lee(x,t-499) = z(t+1)/2 +0.5;
          Lee(x,t-299) = z(t+1);
        end

%         if (t <= 100)
%          x_aix(j) = i;
%          j = j+1;
%          Lee(x,t) = z(t+1)/2 +0.5;

%         if (t == 502)
%          Lee_2(x) = z(t+1)/2 +0.5;
%         end
    end
    x = x+1;
  end

 
a = reshape(Lee',[1,1001*300]);
plot(x_aix,a,'.')

% i=-1:0.002:1;
% plot(i,Lee_2,'。')
      