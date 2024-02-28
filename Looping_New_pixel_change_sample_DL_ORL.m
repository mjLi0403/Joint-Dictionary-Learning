clear all;

m=10;
c=40;
train_num=3;        
test=m-train_num;
start_number=10;
end_number=c*train_num;

folder='orl\orl';
linshi0=imread('orl\orl001.bmp');
[row col]=size(linshi0);

for i=1:c
    for k=1:m
        filename=[folder num2str((i-1)*m+k,'%03d')  '.bmp'];
        image=imread(filename);
        input0(i,k,:,:)=image;
        tempory_image=image;
        
        for mmm=1:row
            for nnn=1:col                
                if mmm==1 || nnn==1 || mmm==row || nnn==col
                   tempory_image2(mmm,nnn)=tempory_image(mmm,nnn);
                else
                    time1=cputime;
                    rand1=rand;
                    if rand1<0.3
                       tempory_image2(mmm,nnn)=tempory_image(mmm,nnn);
                    else
                        nighbor_pixels=tempory_image(mmm-1:mmm+1,nnn-1:nnn+1);
                        nighbor_pixels_vector=reshape(nighbor_pixels,9,1);
                        nighbor_pixels_vector(5)=[];
                        time2=cputime;
                        rand2=rand(int8(time2)); 
                        rand2=rand;
                        rand2_to_int=floor(rand2*8);
                        if  rand2_to_int==8
                            tempory_image2(mmm,nnn)=nighbor_pixels_vector(rand2_to_int)^(1/4);
                        else
                            tempory_image2(mmm,nnn)=nighbor_pixels_vector(rand2_to_int+1);
                        end
                    end
                end                    
            end
        end        
        input2(i,k,:,:)= tempory_image2(:,:);
    end
end
input0=double(input0);
input2=double(input2);

for i=1:c
    for k=1:train_num
        my(:,:)=input2(i,k,:,:);
        my2=my(:);
        ex_data2((i-1)*train_num+k,:)=my2/norm(my2);        
        train_label((i-1)*train_num+k,:)=i;
    end
    
    for k=1:test
        my(:,:)=input2(i,k+train_num,:,:);
        my2=my(:);
        data2((i-1)*test+k,:)=my2/norm(my2);
        
        test_label((i-1)*test+k,:)=i;
    end
    
end

ttt=nchoosek(1:m,train_num);
[daxiao1 daxiao2]=size(ttt);  

rrr=1;
      test_0=1; train_0=1;
      for jj=1:c
        input00(:,:)=input0(jj,:,:);
        for k=1:m
            for n=1:daxiao2
                if k==ttt(rrr,n)
                   tempt0(1,:)=input0(jj,k,:);
                   ex_data(train_0,:)=input0(jj,k,:)/norm(tempt0);
                   train_0=train_0+1;       
                   my_record(n,:)=k;
                end
            end        
                       
        end       
        [ppp qqq]=size(my_record);        
        for k=1:m
            biaoji=0;
            for uuu=1:ppp
                if k==my_record(uuu,:);
                   biaoji=1;
                end             
            end
            
            if biaoji==0
               tempt0(1,:)=input0(jj,k,:);
               data(test_0,:)=input0(jj,k,:)/norm(tempt0);
               test_0=test_0+1; 
            end
        end        
      end          
     n_number=c*train_num;
    useful_train=ex_data;
    useful_train2=ex_data2;
    preserved=inv(useful_train*useful_train'+0.01*eye(n_number))*useful_train;
    preserved2=inv(useful_train2*useful_train2'+0.01*eye(n_number))*useful_train2;
       for j=1:test*c
        shiliang=data(j,:);
        shiliang2=data(j,:);  
        solution=preserved*shiliang';
        solution2=preserved2*shiliang';
        for kk=1:c
            contribution(:,kk)=zeros(row*col,1);
            for hh=1:train_num
                contribution(:,kk)=contribution(:,kk)+solution((kk-1)*train_num+hh)*useful_train((kk-1)*train_num+hh,:)';%原始样本
            end            
        end
        for kk=1:c   
            wucha(kk)=norm(shiliang'-contribution(:,kk));
        end        
        [min_value xx]=min(wucha);
        fen_label(j)=xx; 
        
       for kk=1:c
            contribution(:,kk)=zeros(row*col,1);
            for hh=1:train_num
                contribution(:,kk)=contribution(:,kk)+solution2((kk-1)*train_num+hh)*useful_train2((kk-1)*train_num+hh,:)';%辅助人脸
            end            
        end
        for kk=1:c   
   
            wucha2(kk)=norm(shiliang2'-contribution(:,kk));
        end        
        [min_value yy]=min(wucha2);
        fen_label2(j)=yy;   

n=train_num;
weight=(n/m);
ultimate_wucha=weight*wucha+(1-weight)*wucha2;

        [min_value zz]=min(ultimate_wucha);
        ultimate_fen_label(j)=zz;      
       end

errors=0; errors2=0; ultimate_errors=0;
for i=1:test*c
    
    inte=floor((i-1)/test+1);
    label2(i)=inte;
    if fen_label(i)~=label2(i)
        errors=errors+1;
    end    
 
    if fen_label2(i)~=label2(i)
        errors2=errors2+1;
    end    
 
    if ultimate_fen_label(i)~=label2(i)
        ultimate_errors=ultimate_errors+1;
    end    
end    

errors_ratio=errors/c/test
errors_ratio2=errors2/c/test 
ultimate_errors_ratio=ultimate_errors/c/test

addpath(genpath('.\ksvdbox'));
addpath(genpath('.\OMPbox'));
sparsitythres = 9;
iterations = 10;
iterations4ini =1;
knn=30;

alpha=1e-4;
beta=1e-3;
gam=1e-3;


[numtrain,at]=size(train_label);
[numtest,bt]=size(test_label);
H_test=zeros(c,numtest);
H_train=zeros(c,numtrain);
for j=1:numtrain
   a=train_label(j);
    H_train(a,j)=1;
 end
  for j=1:numtest
    b=test_label(j);
    H_test(b,j)=1;
  end


dictsize=120;
[Dinit,Tinit,Cinit,Q_train,Xinit,D_label] = initialization4LCKSVD(ex_data',H_train,dictsize,iterations4ini,sparsitythres);
[Q]=construct_Q(D_label); 
[D,X,V] = Learn_D_X(Xinit,Dinit,Q,alpha,beta,gam,ex_data,knn,iterations);

 Wx = inv(X*X'+eye(size(X*X')))*X*H_train';
 Wx = Wx';
 Wx=normcols(Wx);

 [yuanshiprediction,yuanshiaccuracy] = classification(D, Wx, data', H_test, sparsitythres);
 fprintf('\n yuanshi DL errors rate is %.04f',1-yuanshiaccuracy);

dictsize=120;
[Dinit,Tinit,Cinit,Q_train,Xinit,D_label] = initialization4LCKSVD(ex_data2',H_train,dictsize,iterations4ini,sparsitythres);
[Q]=construct_Q(D_label); 
[D,X,V] = Learn_D_X(Xinit,Dinit,Q,alpha,beta,gam,ex_data2,knn,iterations);
 Wx = inv(X*X'+eye(size(X*X')))*X*H_train';
 Wx = Wx';
 Wx=normcols(Wx);
 [suijiprediction,suijiaccuracy] = classification(D, Wx, data', H_test, sparsitythres);
 fprintf('\n suijin errors rate is %.04f',1-suijiaccuracy);

dictsize=120;

 hechengtrain_data=zeros(c*train_num*2,size(ex_data,2));
 hechenglabel=zeros(c*train_num*2,1);
for i=1:c
    for j=1:train_num
      hechengtrain_data((i-1)*train_num*2+(j-1)*2+1,:)=ex_data((i-1)*train_num+j,:);
      hechengtrain_data((i-1)*train_num*2+j*2,:)=ex_data2((i-1)*train_num+j,:);
      hechenglabel((i-1)*train_num*2+(j-1)*2+1,:)=i;
      hechenglabel((i-1)*train_num*2+j*2,:)=i;
    end
end

[hechengtrain,at]=size(hechenglabel);
hechengH_train=zeros(c,hechengtrain);

for j=1:hechengtrain
  a=hechenglabel(j);
  hechengH_train(a,j)=1;
end

dictsize=120;
[heDinit,heTinit,heCinit,heQ_train,heXinit,heD_label] = initialization4LCKSVD(hechengtrain_data',hechengH_train,dictsize,iterations4ini,sparsitythres);
[heQ]=construct_Q(heD_label); 
[heD,heX,heV] = Learn_D_X(heXinit,heDinit,heQ,alpha,beta,gam,hechengtrain_data,knn,iterations);
 heWx = inv(heX*heX'+eye(size(heX*heX')))*heX*hechengH_train';
 heWx = heWx';
 heWx=normcols(heWx);
 [hechengprediction,hechengaccuracy] = classification(heD, heWx, data', H_test, sparsitythres);
 fprintf('\n hecheng errors rate is %.04f',1-hechengaccuracy);
