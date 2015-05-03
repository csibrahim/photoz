function [cost, grad, f] = GPVC(theta,m,sigma,X,Y,W,training,validation,joint)

global trainCost;
global validCost;
global best_test;

trainSize = sum(training);
testSize = sum(validation);

d = size(X,2);
k = size(Y,2);

P = reshape(theta(1:d*m),m,d);

Lambda = reshape(theta(d*m+1:d*m+d*d*m)',d,d,m);

dP = zeros(size(P));
dLambda = zeros(size(Lambda));

K = zeros(trainSize,m);
Ktest = zeros(testSize,m);
for i=1:m
    
    Delta = bsxfun(@minus,X(training,:),P(i,:));
    V = Delta*Lambda(:,:,i);
    K(:,i) = exp(-0.5*sum(power(V,2),2));
    
    Delta = bsxfun(@minus,X(validation,:),P(i,:));
    V = Delta*Lambda(:,:,i);
    Ktest(:,i) = exp(-0.5*sum(power(V,2),2));
    
end

if(joint)
    K = [K X(training,:) ones(trainSize,1)];
    Ktest = [Ktest X(validation,:) ones(testSize,1)];
end

KW = bsxfun(@times,K,W(training,:));

I = ones(size(K,2),1);
I(m+1:end) = 0;
I = diag(I);

w = (KW'*K+I*sigma)\(KW'*Y(training,:));

error = K*w-Y(training,:);

errorCost = sum(sum(bsxfun(@times,power(error,2),W(training,:))))/(trainSize*k);

E = 2*(bsxfun(@times,error,W(training,:))*w').*K;

for i=1:m
    
    Delta = bsxfun(@minus,X(training,:),P(i,:));
    V = Delta*Lambda(:,:,i);
    dP(i,:) = dP(i,:)+E(:,i)'*V*Lambda(:,:,i)';
    dLambda(:,:,i) = -bsxfun(@times,Delta,E(:,i))'*V;
    
end

dP = dP/(trainSize*k);
dLambda = dLambda/(trainSize*k);
    
regCost = sum(w(1:m,:).^2)/(trainSize*k);

cost = errorCost+sigma*regCost;

grad = [dP(:);dLambda(:)];

trainCost = sqrt(errorCost);

f = Ktest*w;
error = f-Y(validation,:);

errorCost = sum(sum(bsxfun(@times,power(error,2),W(validation,:))))/(testSize*k);
validCost = sqrt(errorCost);

if(isempty(best_test)||(validCost<=best_test))
    
    best_test  = validCost;
    
    save 'best_theta' 'theta' 'w'; 
    
end

end
