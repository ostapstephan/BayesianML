N=171;
p = 0.5;
a=10;
b=2;
m= 0;
for i=1:N
    j = binornd(1,p);
    m = m+j;
    l = i-m;
end
u_mlb = m/(l+m);
u_cpb = (m + a)/(m+a+l+b);
B_likelihood = gamma(m+l)/(gamma(m)*gamma(l))*u_cpb^m * (1-u_cpb)^l;
B_prior = gamma(a+b)/(gamma(a) * gamma(b))*u_cpb^(a-1)*(1-u_cpb)^(b-1);
post = B_prior * B_likelihood;
u_cpb = (m + a)/(m+a+l+b);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gaussian 
GSum = 0;
u=10;
stdev0 = 10;
sigma02 = stdev0^2;
u0 = 0.1;
g = normrnd(10, 10, [1, N]);
G_prior = ((2*pi*sigma02)^-0.5)*exp(-((u-u0)^2)/(2*sigma02));
stdev = 10;
sigma2 = stdev^2;
sigmaN2 = 1/sigma02 + N/sigma2;
G_Error=zeros(1,N);
SqError=zeros(1,N);
u_mlg=zeros(1,N);
u_N=zeros(1,N);
GSum = 0;
GTotal= zeros(1,N);
for o=1:N
    GSum = GSum + g(1,o);
    GTotal(1,o) = GSum;
u_mlg(1,o) = GTotal(1,o)/o;  
u_N(1,o) = sigma2/(N*sigma02 + sigma2)*u0 + N*sigma2/(N*sigma02 + sigma2)*u_mlg(1,o);
%SqError(1,o)= (u_mlg(1,o) - u_N(1,o))^2;
SqError(1,o)= (u_mlg(1,o) - u_N(1,o))^2;
G_Error(1,o)=sum(SqError)/o;
end
%G_posterior = ((2*pi/sigmaN2)^-0.5)*exp((-(u-u_N)^2)*2*sigmaN2);


%Carena Toy
%Conjugate-Priors
%Due 10/2

close all; clear all; clc;
%BERNOULLI
x = 100; p= 0.7; data = binornd(1,p,1,x);   %generate data
mle=mean(data);                             %maximum likelihood estimate

%Bayesian Estimates
a=3;b=4;av=a/(a+b);                          %GOOD estimates
m=[];
MSEn=0;
N=x;
hs=(1:100);
MSEdata=zeros(1,100);
MSEplot=zeros(2,100);

u=linspace(0,1,101);    %for plotting 2x2
prior=zeros(1,101); %for prior in 2x2
for i=1:101
prior(1,i)=(gamma(a+b)*(u(1,i)^(a-1))*(1-u(1,i))^(b-1))/(gamma(a)*gamma(b));
end

posterior=zeros(1,101);
for h = 1:N
    m=sum(data(1:h));
    l=h-m;
    frac=(m+a)/(m+a+l+b); %Bayesian estimate
    
    mse0=(frac-mle)^2;  %mean squared estimate
    MSEo=MSEn;
    MSEn=MSEo+mse0;
    MSE=MSEn/h;
    MSEdata(h)=MSE;
    
    if h==33        %for 2nd subplot
        subplot2data=[];
        for i=1:101
            posterior(1,i)=gamma(a+b+l+m)*((u(1,i))^(m+a-1))*((1-u(1,i))^(l+b-1))/(gamma(m+a)*gamma(l+b));
        end
        subplot2data=posterior;
    elseif h==67        %for 3rd subplot
        subplot3data=[];
        for i = 1:101
            posterior(1,i)=gamma(a+b+l+m)*((u(1,i))^(m+a-1))*((1-u(1,i))^(l+b-1))/(gamma(m+a)*gamma(l+b));
        end
        subplot3data=posterior;
    elseif h==100       %final subplot
        subplot4data=[];
        for i=1:101
            posterior(1,i)=gamma(a+b+l+m)*((u(1,i))^(m+a-1))*((1-u(1,i))^(l+b-1))/(gamma(m+a)*gamma(l+b)); 
        end 
        subplot4data=posterior;
    end
end

c=3;d=14;avb=c/(c+d);       %BAD estimates
MSEnb=0; MSEdatab=zeros(1,100);
for h = 1:N
    m=sum(data(1:h));
    l=h-m;
    frac=(m+c)/(m+c+l+d);
    
    mse0b=(frac-mle)^2;
    MSEob=MSEnb;
    MSEnb=MSEob+mse0b;
    MSEb=MSEnb/h;

    MSEdatab(h)=MSEb;
end
figure(1);
plot(hs,MSEdata,hs,MSEdatab);

figure(2);
%subplots of prior and posterior densities
subplot(2,2,1)
plot(u,prior);
subplot(2,2,2)
plot(u,subplot2data);
subplot(2,2,3)
plot(u,subplot3data);
subplot(2,2,4)
plot(u,subplot4data);

%GAUSSIAN
mu=10;var=3;sze=100;sigma=sqrt(var);
gdata=normrnd(mu,sigma,sze);
%priors info
mug0=9; %good
varg0=2;
mub0=1; %bad
varb0=0.5;

mseg=[];mur=[];
mugbay=[];mubbay=[];
msegbay=[];msebbay=[];
mseml=[];mleold=[];

mleg=mean(gdata, 'all');
U=linspace(0,20,101);    %for plotting 2x2
priorG=zeros(1,101);     %for prior in 2x2
for i=1:101
priorG(1,i)=exp(-(U(1,i)-mug0)^2/(2*varg0))/(sqrt(varg0)*sqrt(2*pi()));
end
%GOOD MSEs & density calculations
msei=0;
mse_current = 0;
MSEGgdata=zeros(1,100);
posteriorG=zeros(1,101);
for i=1:sze
    mu_g_i=mean(gdata(1:i));
    
    mu_g_g=(var*mug0/(i*varg0+var))+i*varg0*mu_g_i/(i*varg0+var);
    var_g_g=1/((1/varg0)+(i/var));
    
    
    msei=(mleg-mu_g_g)^2;       %Mean squared error calculations
    mse_old=mse_current;
    mse_current=mse_old+msei;
    MSEg=mse_current/i;
    MSEGgdata(i)=MSEg;
    
    %density function
    posteriorG=zeros(1,101);
    %posteriorG(1,i)=exp(-0.5*((U(1,i)-mu_g_g)/sqrt(var_g_g))^2)/sqrt(var_g_g*2*pi());
    if i ==33
        subplotdata2=zeros(1,101);
        posteriorG=normpdf(U,mu_g_g,sqrt(var_g_g));
        %for h=1:101
            %mssesess=U(1,h);
            %fprintf('%d\n',mssesess);
            %posteriorG(1,h)=normpdf(U(1,h),mu_g_g,var_g_g);
            
            %posteriorG(1,h)=gaussmf(U(1,h),[sqrt(var_g_g),mu_g_g]);
            %posteriorG(1,h)=exp(-((U(1,h))-mu_g_g)^2/(2*var_g_g))/(sqrt(var_g_g*2*pi()));
            fprintf('%d\n',posteriorG);   
        %end
        subplotdata2=posteriorG;
    elseif i ==67
        subplotdata3=zeros(1,101);
        %fprintf('elseif for plot 3 working\n');
        %for h=1:101
            %fprintf('for loop 3 working\n');
            %posteriorG(1,h)=exp(-((U(1,h))-mu_g_g)^2/(2*var_g_g))/(sqrt(var_g_g)*sqrt(2*pi()));
            %posteriorG=gaussmf(U(1,h),[sqrt(var_g_g),mu_g_g]);
            %fprintf('%d,    %d,      %d\n',sqrt(var_g_g), mu_g_g, posteriorG);    
        %end
        posteriorG=normpdf(U,mu_g_g,sqrt(var_g_g));
        subplotdata3=posteriorG;
    elseif i ==sze
        subplotdata4=zeros(1,101);
%        fprintf('elseif for plot 4 working\n');
        posteriorG=normpdf(U,mu_g_g,sqrt(var_g_g));
        %for h=1:101
          %  fprintf('for loop 4 working\n');
            %posteriorG(1,h)=exp(-((U(1,h))-mu_g_g)^2/(2*var_g_g))/(sqrt(var_g_g)*sqrt(2*pi()));
         %   posteriorG(1,h)=gaussmf(U(1,h),[sqrt(var_g_g),mu_g_g]);
        %    fprintf('%d\n',posteriorG);   
       % end
        subplotdata4=posteriorG;
    end  
end
%BAD MSE calculation

mseib=0;
mse_currentb = 0;
MSEGbdata=zeros(1,100);
for i=1:sze
    mu_b_i=mean(gdata(1:i));
    
    mu_b_g=(var*mub0/(i*varb0+var))+i*varb0*mu_b_i/(i*varb0+var);
    
    mseib=(mleg-mu_b_g)^2;
    mse_oldb=mse_currentb;
    mse_currentb=mse_oldb+mseib;
    MSEb=mse_currentb/i;

    MSEGbdata(i)=MSEb;
end

%Plotting for mse and densities of Gaussians
figure(3);
plot(hs,MSEGgdata,hs,MSEGbdata);

figure(4);
subplot(2,2,1);
plot(U,priorG);
subplot(2,2,2);
plot(U,subplotdata2);
subplot(2,2,3);
plot(U,subplotdata3);
subplot(2,2,4);
plot(U,subplotdata4);

