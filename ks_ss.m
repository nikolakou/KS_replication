%% density approach: r0 = 0.1050; 10 seconds


clear all;
clc;
tic;


M  = 500;                        % number of asset grid points
aM = 110;                         % maximum asset level
A  = linspace(0,aM,M)' ; 

%discretize income process

Markov.autocorr = 0.9;
Markov.num_states = 7.0;
Markov.mean = 0;
Markov.sd = 0.10/(1-Markov.autocorr^2);

[transition_matrix, income_states] = rouwen(Markov.autocorr,Markov.mean,Markov.sd,Markov.num_states);

[Amat,Ymat] = ndgrid(A,exp(income_states));


%Param structure

Param.gamma = 2
Param.beta = 0.95
Param.alpha = 1/3
Param.deprec = 0.08

%I and T not needed here (I think)
crit=10^(-6)
I = 10^4
T=10^4
r0 = 0.1051

%%

max_iterr=10;
dist=1;
iterr=1;

tic;

while dist>0.0001 && iterr<max_iterr
    
policyfun = policy_fun(r0,crit,I,T,Amat,Ymat,Param,transition_matrix,income_states);

next = interp1(A,A, policyfun, "next");
previous = interp1(A,A, policyfun, "previous");

next(isnan(next)) = aM;
previous(isnan(previous))=aM;

Q= zeros(length(A),length(A),7);

for k=1:length(A)
    for j = 1:length(Ymat(1,:))
        if next(k,j)==previous(k,j)
            Q(A==previous(k,j),k,j)=1;
        else
Q(A==previous(k,j),k,j) = (next(k,j) - policyfun(k,j))./(next(k,j)-previous(k,j));
Q(A==next(k,j),k,j) = (policyfun(k,j)-previous(k,j))./(next(k,j)-previous(k,j));
        end
    end
end

big_transition = zeros(length(A)*length(Ymat(1,:)));


for j=1:length(Ymat(1,:))
for i=1:length(Ymat(1,:))
big_transition((i-1)*length(A)+1:i*length(A),(j-1)*length(A)+1:j*length(A))=Q(:,:,j).*transition_matrix(j,i);
end
end

big_transition = big_transition';

% for j= 1:length(big_transition)
% big_transition(j,:)=big_transition(j,:)/sum(big_transition(j,:));
% end

%%

initial = ones(length(A)*length(Ymat(1,:)),1)'.*(1/(length(A)*length(Ymat(1,:))));

dist = 1;
max_iter = 1000;
iter=1;

while dist>1e-17 && iter<max_iter
next_initial = initial*big_transition;
dist = norm(next_initial - initial);
initial = next_initial;
iter=iter+1;
end

%%

% capital=zeros(length(Ymat(1,:)),1);
% for j=1:length(Ymat(1,:))
% capital(j,1)=A'*initial((j-1)*(length(A))+1:j*length(A));
% end

% capital=zeros(length(Ymat(1,:)),1);
% 
% for j=1:length(Ymat(1,:))
% capital(j,1)=A'*initial((j-1)*(length(A))+1:j*length(A));
% end
%%

stationary_it = zeros(length(A),1);

for l= 1:length(A)
   stationary_it(l,1)=sum(initial(l:length(A):end));
end
%%

agg_capital = A'*stationary_it;

rsupply = Param.alpha*(agg_capital)^(Param.alpha-1);

r0 = 0.8*r0 + 0.2*(rsupply)

dist = abs(r0-rsupply)

iterr=iterr+1


end

toc;

function [policyfun] = policy_fun(r0,crit,I,T,Amat,Ymat,Param,transition_matrix,income_states)

fprintf("Start solving Aiyagari...")

%functions here now

% (inverse) marginal utility functions
up    = @(c) c.^(-Param.gamma);        % marginal utility of consumption
invup = @(x) x.^(-1/Param.gamma);      % inverse of marginal utility of consumption 

% current consumption level, cp0(anext,ynext) is the guess
C0 = @(cp0,r) invup(Param.beta*(1+r-Param.deprec)*up(cp0)*transition_matrix');
                
% current asset level, c0 = C0(cp0(anext,ynext))
A0 = @(anext,y,c0,r,w) 1/(1+r-Param.deprec)...
                    *(c0+anext-y.*w);
                
%%%%

[M,N]=size(Amat);

w0 = (1-Param.alpha)*(Param.alpha./(r0))^(Param.alpha/(1-Param.alpha));

cp0 = (r0-Param.deprec)*Amat + Ymat*w0;

dist=crit+1;

maxiter=10^3;

iter = 1;

fprintf("inner loop, running... \n");

while (dist>crit && iter<maxiter)
    
c0 = C0(cp0,r0);

a0=A0(Amat,Ymat,c0,r0,w0);

cpbind = (1+r0-Param.deprec).*Amat + Ymat*w0;

cpnon = zeros(M,N);

for j=1:N
    
    cpnon(:,j)=interp1(a0(:,j),c0(:,j),Amat(:,j),"spline");
    
end

for j = 1:N
cpnext(:,j)=(Amat(:,j)>a0(1,j)).*cpnon(:,j)+(Amat(:,j)<=a0(1,j)).*cpbind(:,j);
end

if mod(iter,100) ==1
    fprintf("inner loop, iteration: %3i, Norm: %2.6f \n",[iter,dist]);
end

dist = norm((cpnext-cp0),Inf)

iter = iter+1;

cp0 = cpnext;


end

policyfun = (1+r0-Param.deprec)*Amat+w0*Ymat - cp0;

end

function [P_Rouw, z_Rouw] = rouwen(rho_Rouw, mu_uncond, sig_uncond, n_R)
%ROUWEN   Rouwenhorst's method (1995) to approximate an AR(1) process using 
%   a  finite state Markov process. 
%
%   For details, see Rouwenhorst, G., 1995: Asset pricing  implications of 
%   equilibrium business cycle models, in Thomas Cooley (ed.), Frontiers of 
%   Business Cycle Research, Princeton University Press, Princeton, NJ.
% 
%   Suppose we need to approximate the following AR(1) process:
%
%                   y'=rho_Rouw*y+e
%
%   where abs(rho_Rouw)<1, sig_uncond=std(e)/sqrt(1-rho_Rouw^2) and 
%   mu_uncond denotes E(y), the unconditional mean of y. Let n_R be the 
%   number of grid points. n_R must be a positive integer greater than one.  
%
%   [P_Rouw, z_Rouw] = rouwen(rho_Rouw, mu_uncond, sig_uncond, n_R) returns  
%   the discrete state space of n_R grid points for y, z_Rouw, and 
%   the centrosymmetric transition matrix P_Rouw. Note that
%
%       1. z_Rouw is a column vector of n_R real numbers. 
%       2. The (i,j)-th element of P_Rouw is the conditional probability 
%          Prob(y'=z_Rouw(i)|y=z_Rouw(j)), i.e.
%
%                 P_Rouw(i,j)=Prob(y'=z_Rouw(i)|y=z_Rouw(j))
%
%           where z_i is the i-th element of vector z_Rouw. Therefore 
%
%           P_Rouw(1,j)+P_Rouw(2,j)+ ... +P_Rouw(n,j)=1 for all j.
%   
%   See also HITM_Z and HITM_S on how to simulate a Markov processes using 
%   a transition matrix and the grids. 
%
%   Damba Lkhagvasuren, June 2005

% CHECK IF abs(rho)<=1 
if abs(rho_Rouw)>1
    error('The persistence parameter, rho, must be less than one in absolute value.');
end

% CHECK IF n_R IS AN INTEGER GREATER THAN ONE.
if n_R <1.50001 %| mod(n_R,1)~=0 
    error('For the method to work, the number of grid points (n_R) must be an integer greater than one.');  
end

% CHECK IF n_R IS AN INTEGER.
if mod(n_R,1)~=0 
    warning('the number of the grid points passed to ROUWEN is not an integer. The method rounded n_R to its nearest integer.')
    n_R=round(n_R);
    disp('n_R=');
    disp(n_R);  
end

% GRIDS
step_R = sig_uncond*sqrt(n_R - 1); 
z_Rouw=[-1:2/(n_R-1):1]';
z_Rouw=mu_uncond+step_R*z_Rouw;

% CONSTRUCTION OF THE TRANSITION PROBABILITY MATRIX
p=(rho_Rouw + 1)/2;
q=p;

P_Rouw=[ p  (1-p);
        (1-q) q];
    
    for i_R=2:n_R-1
    a1R=[P_Rouw zeros(i_R, 1); zeros(1, i_R+1)];
    a2R=[zeros(i_R, 1) P_Rouw; zeros(1, i_R+1)];
    a3R=[zeros(1,i_R+1); P_Rouw zeros(i_R,1)];
    a4R=[zeros(1,i_R+1); zeros(i_R,1) P_Rouw];
    P_Rouw=p*a1R+(1-p)*a2R+(1-q)*a3R+q*a4R;
    P_Rouw(2:i_R, :) = P_Rouw(2:i_R, :)/2;
    end
    
P_Rouw=P_Rouw';

for i_R = 1:n_R
    P_Rouw(:,i_R) = P_Rouw(:,i_R)/sum(P_Rouw(:,i_R));
end
end


