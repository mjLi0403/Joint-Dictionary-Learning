function [Dinit,Tinit,Winit,Q,Xtemp,dictLabel]=initialization4LCKSVD(training_feats,H_train,dictsize,iterations,sparsitythres)

numClass = size(H_train,1); 
numPerClass = round(dictsize/numClass); 
Dinit = []; 
dictLabel = [];
for classid=1:numClass
    col_ids = find(H_train(classid,:)==1);
    data_ids = find(colnorms_squared_new(training_feats(:,col_ids)) > 1e-6);
    perm = [1:length(data_ids)];
    Dpart = training_feats(:,col_ids(data_ids(perm(1:numPerClass))));
    para.data = training_feats(:,col_ids(data_ids));
    para.Tdata = sparsitythres;
    para.iternum = iterations;
    para.memusage = 'high';
    para.initdict = normcols(Dpart);
    [Dpart,Xpart,Errpart] = ksvd(para,'');
    Dinit = [Dinit Dpart];
    
    labelvector = zeros(numClass,1);
    labelvector(classid) = 1;
    dictLabel = [dictLabel repmat(labelvector,1,numPerClass)];
end


T = eye(dictsize,dictsize);
Q = zeros(dictsize,size(training_feats,2));
for frameid=1:size(training_feats,2)
    label_training = H_train(:,frameid);
    [maxv1,maxid1] = max(label_training);
    for itemid=1:size(Dinit,2)
        label_item = dictLabel(:,itemid);
        [maxv2,maxid2] = max(label_item);
        if(maxid1==maxid2)
            Q(itemid,frameid) = 1;
        else
            Q(itemid,frameid) = 0;
        end
    end
end

params.data = training_feats;
params.Tdata = sparsitythres;
params.iternum = iterations;
params.memusage = 'high';

params.initdict = normcols(Dinit);

[Dtemp,Xtemp,Errtemp] = ksvd(params,'');

Winit = inv(Xtemp*Xtemp'+eye(size(Xtemp*Xtemp')))*Xtemp*H_train';
Winit = Winit';

Tinit = inv(Xtemp*Xtemp'+eye(size(Xtemp*Xtemp')))*Xtemp*Q';
Tinit = Tinit';
