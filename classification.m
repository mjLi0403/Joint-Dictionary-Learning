function [prediction, accuracy, err] = classification(D, W, data, Hlabel, sparsity)

G = D'*D;
Gamma = omp(D'*data,G,sparsity);


errnum = 0;
err = [];
prediction = [];
for featureid=1:size(data,2)
    spcode = Gamma(:,featureid);
    score_est =  W * spcode;
    score_gt = Hlabel(:,featureid);
    [maxv_est, maxind_est] = max(score_est);
    [maxv_gt, maxind_gt] = max(score_gt);
    prediction = [prediction maxind_est];
    if(maxind_est~=maxind_gt)
        errnum = errnum + 1;
        err = [err;errnum featureid maxind_gt maxind_est];
    end
end
accuracy = (size(data,2)-errnum)/size(data,2);