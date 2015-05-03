function [ cost, grad] = GPVL(theta,m,sigma,X,Y,W,training,validation,joint)

% setting global variables to keep track of best performing models

global trainCost;
global validCost;
global best_test;

%%%%%%%%%%%%%% EXTRACT-PARAMETERS %%%%%%%%%%%%%%%%

trainSize = sum(training);
validSize = sum(validation);

d = size(X,2);
k = size(Y,2);

P = reshape(theta(1:d*m),m,d);

lambda = theta(d*m+1:d*m+m)';

dP = zeros(size(P));
dlambda = zeros(size(lambda));

D = Dxy(X(training,:),P);

K = exp(bsxfun(@rdivide,D,-2*lambda.^2));

if(joint)
    K = [K X(training,:) ones(size(K,1),1)];
end

KW = bsxfun(@times,K,W(training,:));

I = ones(size(K,2),1);
I(m+1:end) = 0;
I = diag(I);

w = (KW'*K+I*sigma)\(KW'*Y(training,:));

f = K*w;

error = f-Y(training,:);

errorCost = sum(sum(bsxfun(@times,power(error,2),W(training,:))))/(trainSize*k);


for j=1:m
    E = 2*(bsxfun(@times,error,W(training,:))*w(j,:)').*K(:,j);
    Delta = bsxfun(@minus,X(training,:),P(j,:));
    dP(j,:) = E'*Delta/lambda(j).^2;    
    dlambda(j) = E'*D(:,j)/lambda(j).^3;
end

dP = dP/(trainSize*k);
dlambda = dlambda/(trainSize*k);

regCost = sigma*sum(sum(w(1:m,:).^2))/(trainSize*k);
cost = errorCost+regCost;

grad = [dP(:);dlambda(:)];

trainCost = sqrt(errorCost);

D = Dxy(X(validation,:),P);

K = exp(bsxfun(@rdivide,D,-2*lambda.^2));

if(joint)
    K = [K X(validation,:) ones(size(K,1),1)];
end

f = K*w;
error = f-Y(validation,:);
errorCost = sum(sum(bsxfun(@times,power(error,2),W(validation,:))))/(validSize*k);

validCost = sqrt(errorCost);

if(isempty(best_test)||(validCost<=best_test))
    
    best_test  = validCost;
    
    save 'best_theta' 'theta' 'w'; 
    
end

end
