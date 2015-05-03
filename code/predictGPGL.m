function f = predictGPGL(theta,m,w,joint,X)


[n,d] = size(X);

P = reshape(theta(1:d*m),m,d);

lambda = theta(m*d+1)';

D = Dxy(X,P);

K = exp(-0.5*D./lambda^2);

if(joint)
    K = [K X ones(size(n,1),1)];
end

f = K*w;

end
