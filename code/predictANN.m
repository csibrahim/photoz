function f = predictANN(theta,m,k,joint,X)


[n,d] = size(X);

dim = m;
if(joint)
    dim = m+d+1;
end

W1 = reshape(theta(1:d*m),d,m);
W2 = reshape(theta(d*m+1:d*m+dim*k),dim,k);

b1 = reshape(theta(d*m+dim*k+1:d*m+dim*k+m),1,m);
b2 = reshape(theta(d*m+dim*k+m+1:end),1,k);


f1 = tanh(bsxfun(@plus, X*W1,b1));
if(joint)
    f1 = [f1 X ones(n,1)];
end

f = bsxfun(@plus,f1*W2,b2);

end
