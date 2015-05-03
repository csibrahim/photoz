function f = predictGPVC(theta,m,w,joint,X)

[n,d] = size(X);

P = reshape(theta(1:d*m),m,d);

Lambda = reshape(theta(d*m+1:d*m+d*d*m)',d,d,m);

K = zeros(n,m);

for i=1:m
    
    Delta = bsxfun(@minus,X,P(i,:));
    V = Delta*Lambda(:,:,i);
    K(:,i) = exp(-0.5*sum(power(V,2),2));
    
end

if(joint)
    K = [K X ones(n,1)];
end

f = K*w;

end
