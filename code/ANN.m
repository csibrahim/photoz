function [ cost, grad, f2] = ANN(theta,m,sigma,X,Y,W,training,validation,joint)

% setting global variables to keep track of best performing models

global trainCost;
global validCost;
global best_test;

%%%%%%%%%%%%%% EXTRACT-PARAMETERS %%%%%%%%%%%%%%%%

trainSize = sum(training);
testSize = sum(validation);

d = size(X,2);
k = size(Y,2);

dim = m;
if(joint)
    dim = m+d+1;
end
W1 = reshape(theta(1:d*m),d,m);
W2 = reshape(theta(d*m+1:d*m+dim*k),dim,k);

b1 = reshape(theta(d*m+dim*k+1:d*m+dim*k+m),1,m);
b2 = reshape(theta(d*m+dim*k+m+1:end),1,k);

%%%%%%%%%%%%%% FEED-FORWARD %%%%%%%%%%%%%%%%%%%%%%

f1 = tanh(bsxfun(@plus, X(training,:)*W1,b1));
if(joint)
    f1 = [f1 X(training,:) ones(size(f1,1),1)];
end
f2 = bsxfun(@plus,f1*W2,b2);

%%%%%%%%%%%%%% BACK-PROPAGATION %%%%%%%%%%%%%%%%%%

error = f2-Y(training,:);
errorCost = sum(sum(bsxfun(@times,error.^2,W(training,:))))/(trainSize*k);

delta2 = 2*bsxfun(@times,error,W(training,:));

dW2 = (f1'*delta2)+2*sigma*W2;
db2 = sum(delta2);

delta1 = (delta2*W2').*(1-f1.^2);

dW1 = X(training,:)'*delta1(:,1:m)+2*sigma*W1;
db1 = sum(delta1(:,1:m));

%%%%%%%%%%%%%%% RETURN VALUES %%%%%%%%%%%%%%%%%%%

grad = [dW1(:);dW2(:);db1(:);db2(:)]/(trainSize*k);
regCost = (sum(W1(:).^2)+sum(W2(:).^2))/(trainSize*k);
cost = errorCost+sigma*regCost;

trainCost = sqrt(errorCost);

f1 = tanh(bsxfun(@plus, X(validation,:)*W1,b1));
if(joint)
    f1 = [f1 X(validation,:) ones(size(f1,1),1)];
end
f2 = bsxfun(@plus,f1*W2,b2);

error = f2-Y(validation,:);
errorCost = sum(sum(bsxfun(@times,error.^2,W(validation,:))))/(testSize*k);
validCost = sqrt(errorCost);

if(isempty(best_test)||(validCost<=best_test))
    
    best_test  = validCost;
    
    save 'best_theta' 'theta'; 
    
end

end
