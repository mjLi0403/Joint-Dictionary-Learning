function [D,X,V] = Learn_D_X(Xinit,Dinit,Q,alpha,beta,gam,train_data,kn,diedai)


 X=Xinit;
 D=Dinit;
[Q_size,Q_size]=size(Q);


options = [];
options.NeighborMode = 'KNN';
options.k = kn;
options.WeightMode = 'HeatKernel';


for j=1:diedai
    
[L]=construct_L(D,options);
                   
V1=(D'*D+Q*beta+gam*eye(Q_size));
[numv1,av1]=size(find(isnan(V1)));
[numv2,av2]= size(find(isinf(V1)));
if(numv1>0 )||(numv2>0)
V1=eye(Q_size);
end
V3=pinv(V1);
V2=(gam*X+D'*train_data');
V=V3*V2;


D1=(X*X'+V*V');
[numd1,ad1]=size(find(isnan(D1)));
[numd2,ad2]= size(find(isinf(D1)));
if(numd1>0 )||(numd2>0)
D1=eye(Q_size);
end
D3=pinv(D1);
D2=train_data'*(X'+V');
D=D2*D3;

X1=(D'*D+alpha*L+gam*eye(Q_size));
[numx1,ax1]=size( find(isnan(X1)));
[numx2,ax2]= size(find(isinf(X1)));
 if(numx1>0 )||(numx2>0)
    X1=eye(Q_size);
 end
X3=pinv(X1);

X2=(gam*V+D'*train_data');
X=X3*X2;
 
end


D=normcols(D);
X=normcols(X);
V=normcols(V);
                

