function f = predictGPVL(theta,m,w,joint,X)


[n,d] = size(X);

P = reshape(theta(1:d*m),m,d);

lambda = theta(d*m+1:d*m+m)';

D = Dxy(X,P);

K = exp(bsxfun(@rdivide,D,-2*lambda.^2));

if(joint)
    K = [K X ones(n,1)];
end

f = K*w;

end
