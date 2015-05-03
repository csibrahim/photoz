
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataPath = 'EUCLID_MAG';    % path to data, should load two variables X and Y.
                            % X should have the first half the columns as the magnitudes and the rest as the corresponding error bars.
                            % Y should have the spectroscopic redshift for
                            % the corresponding sources in X.

displayPlots = true;        % displays the scatter and the box plot if set to true

m = 10;                     % number of basis functions

method = 4;                 % 1=ANN, 2=GPGL, 3=GPVL, 4=GPVC
prior = 2;                  % 0 = Zero, 1=Linear, 2=Joint

sigma = 1;                  % regularization parameter

balanced = true;            % setting to true will balance the weights to account for the output bias
normalised = true;          % setting to true optimises (Zs-Zp)/(1+Zs)
decorrelate = true;         % set to true if decorrelation is desired
removeError = true;         % set to true to ignore the magnitude errors when training

filterID = 5;               % the filter id to base the cut-off on
filterCut = 100;            % the cut-off value, set to high number if not desired, e.g 100

lowerBound = 0.2;           % redshift lower bound
upperBound = 2;             % redshift upper bound

validSplit = 0.1;           % validation set percentage
testSplit  = 0.1;           % test set percentage
trainSplit = 0.8;           % training set percentage.If the sum do not add to 1 with the validation and test splits, 
                            % the rest will just be ignored


interval = 0.01;            % interval used to group the data into bins for weight assignment

maxIter = 500;              % maximum number of iterations

seed = 1;                   % random seed

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath minFunc

clear global;

options.method = 'lbfgs';
options.maxIter = maxIter;

rand('seed',seed);

load(dataPath);

outOfRange = Y>upperBound|Y<lowerBound;

X(outOfRange,:) = [];
Y(outOfRange,:) = [];

[n,d] = size(X);
k = size(Y,2);

filters = d/2;

if(removeError)
    X(:,filters+1:end) = [];
    d = filters;
else
    X(:,filters+1:end) = log(X(:,filters+1:end));
end

ltCut = X(:,filterID)<filterCut;
gtCut = X(:,filterID)>=filterCut;

W = ones(size(Y));

if(balanced)
    
    bins = (lowerBound:interval:upperBound)';
    counts = hist(Y,bins);
    [~,ind] = min(Dxy(Y,bins),[],2);
    W = max(counts)./counts(ind)';
end

if(normalised)
    W = W./(power(1+Y,2));
end

r = randperm(n);

validSize = ceil(n*validSplit);
testSize  = ceil(n*testSet);
trainSize = min(ceil(trainSplit*n),n-testSize-validSize);

validation = false(n,1);
testing  = false(n,1);
training = false(n,1);

validation(r(1:validSize)) = true;
testing(r(validSize+1:validSize+testSize)) = true;
training(r(validSize+testSize+1:validSize+testSize+trainSize))=true;

training = training&ltCut;
validation = validation&ltCut;

if(decorrelate)
    [muX,T,Ti] = pca(X(training,:),1);
    X = bsxfun(@minus,X,muX)*T;
end

if(prior==1)
    A = [X ones(size(X,1),1)];

    AW = bsxfun(@times,A,W);
    
    w = (AW(training,:)'*A(training,:))\(AW(training,:)'*Y(training,:));

    Yl = A*w;

    Y = Y-Yl;
end

if(method>1)
    [~,P] = kmeans(X(training,:),m,'options',statset('Display','iter','Maxiter',50));
end

if(method==1)
    
    dim = m;
    if(prior==2)
        dim = m+d+1;
    end
    
    W1 = zscore(rand(d,m))/sqrt(d);
    b1 = zeros(m, 1);

    W2 = zscore(rand(dim,k))/sqrt(m);
    b2 = zeros(k, 1);

    theta = [W1(:);b1(:);W2(:);b2(:)];
    trainFun = @(params) ANN(params,m,sigma,X,Y,W,training,validation,prior==2);
    
    theta = minFunc(trainFun,theta,options);
    load 'best_theta';
    
    Yp = predictANN(theta,m,k,prior==2,X);
    
elseif(method==2)
    
    theta = [P(:);1];
    trainFun = @(params) GPGL(params,m,sigma,X,Y,W,training,validation,prior==2);
    
    theta = minFunc(trainFun,theta,options);
    load 'best_theta';
    
    Yp = predictGPGL(theta,m,w,prior==2,X);
    
elseif(method==3)
    
    theta = [P(:);ones(m,1)];
    trainFun = @(params) GPVL(params,m,sigma,X,Y,W,training,validation,prior==2);
    
    theta = minFunc(trainFun,theta,options);
    load 'best_theta';
    
    Yp = predictGPVL(theta,m,w,prior==2,X);
    
else
    
    Lambda = zeros(d,d,m);

    for j=1:m
        Lambda(:,:,j) = eye(d);
    end
    
    theta = [P(:);Lambda(:)];
    trainFun = @(params) GPVC(params,m,sigma,X,Y,W,training,validation,prior==2);
    
    theta = minFunc(trainFun,theta,options);
    load 'best_theta';
    
    Yp = predictGPVC(theta,m,w,prior==2,X);
end


if(prior==1)
    Y = Y+Yl;
    Yp = Yp+Yl;
end


fprintf('Full\n');
displayResults(Y,Yp,testing,displayPlots);

if(any(gtCut)>0&&any(ltCut))
    fprintf('\n');
    fprintf('Filter < cut-off\n');
    displayResults(Y,Yp,testing&ltCut,false);
    fprintf('\n');
    fprintf('Filter >= cut-off\n');
    displayResults(Y,Yp,testing&gtCut,false);
end
