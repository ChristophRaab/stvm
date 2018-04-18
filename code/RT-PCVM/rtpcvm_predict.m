function [erate, nvec, y_sign, y_prob] = rtpcvm_predict(testY,model)

sizeM = size(model,2);

if sizeM == 1
    m = size(model.trainY,1);
   
    K = model.K(m+1:end, 1:m);
    w = model.w;
    b = model.b;
    trainY = model.trainY;
    used = model.used;
    
    weights = w(used).*trainY(used);
    
    % Compute RVM over test data and calculate error
    PHI	= K(:,used);
    test_num = size(K,1);
    
    Y_regress = PHI*weights+b*ones(test_num,1);
    y_sign	= sign(Y_regress);
    
    % the probablistic output
    y_prob = normcdf(Y_regress);
    
    errs	= sum(y_sign(testY== -1)~=-1) + sum(y_sign(testY==1)~=1);
    erate = errs/test_num;
    nvec = length(used);
elseif sizeM > 2
    fprintf('\nMulticlass Prediction detected! Splitting up test data..\n')
    usedVectors = [];
    multiLabels = [];
    multiProb = [];
    %     multiProb = zeros(size(testY,1),sizeM);
    
    % For-Loop to calculate One vs One prediction
    for i = 1:size(model,2)
        trainY = model(i).trainY;
        % Taking the corrosponding labels from original train label vector
        oneIndx = find(trainY == model(i).one);
        twoIndx = find(trainY == model(i).two);
        
        % Merge the label vectors into one training vector
        trainYOR = [ones(size(oneIndx,1),1); ones(size(twoIndx,1),1)*-1];
        
        m = size(trainYOR,1);
        
        % Taking the lower left square for prediction
    
        [erate, nvec, label, y_prob] = rtpcvm_predict(testY,model(i));
        
        label(find(label==1)) = model(i).one;
        label(find(label==-1)) = model(i).two;
        
        multiLabels = [multiLabels label];
        multiProb = [multiProb y_prob];
        
        vectors =model(i).used;
        
        usedVectors = [usedVectors; vectors];
        
    end
    
     
    multiLabels(multiLabels==0) = NaN;
    
    
    multiProb(multiProb==0) = NaN;
    
    % Untrustable Result starts here:
    % Get most frequent labels for observation
    [y_sign,F,C] = mode(multiLabels,2);
    % Find ties of label frequency
    indx = find( cellfun(@(V) any(isnan(V(:))), C) == 1);
    % Get index from the Prob estimate
    cdf = multiProb(indx,:);
    % Get the max prob from above tie
    [M,I] = max(cdf');
    indxMultiLabel = [indx,I'];
    
    % Override label frome mode with that labels which has the highest prob
    for i=1:size(indxMultiLabel,1)
        y_sign(indxMultiLabel(i,1)) = multiLabels(indxMultiLabel(i,1),indxMultiLabel(i,2));
    end
    %----
    % Prob estimate by  mean... TODO: needs to be changed
    y_prob = nanmean(multiProb')';
    
    
    % Take rounded means to for the integer class label assignment
    % y_sign = round(nanmean(multiLabels')');
    % nvec = unique(usedVectors);
    resulterror = abs(testY-y_sign);
    erate = size(resulterror(resulterror ~=0),1) / size(testY,1);
    
    fprintf('\nPCTKVM Acc: %f\n',1-erate);
else
    fprintf('\nWrong Input of Models\n');
end


