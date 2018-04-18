% Domain Invariant Transfer Kernel Learning
% M. Long, J. Wang, J. Sun, and P.S. Yu
% IEEE Transactions on Knowledge and Data Engineering (TKDE)

% Contact: Mingsheng Long (longmingsheng@gmail.com)

addpath(genpath('../libsvm/matlab'));
clear all;

domain_type = 'text';       % domain_type: 'text' | 'image'

switch domain_type
    case 'text'
        options.ker = 'linear';     % kernel: 'linear' | 'rbf' | 'lap'
        options.eta = 2.0;          % eigenspectrum damping factor
        svmc = 1.0;                 % SVM complexity regularizer in LibSVM
        
        result = [];
        for strData = {'org_vs_people', 'org_vs_place', 'people_vs_place'}
            for iData = 1:2
                data = char(strData);
                data = strcat(data, '_', num2str(iData));
                load(strcat('../data/', data));
                
                fprintf('data=%s\n', data);
                Xs = bsxfun(@rdivide, Xs, sqrt(sum(Xs.^2, 1)) + eps);
                Xt = bsxfun(@rdivide, Xt, sqrt(sum(Xt.^2, 1)) + eps);
                m = size(Xs, 2);
                n = size(Xt, 2);
                
                K = kernel(options.ker, [Xs, Xt], []);
                model = svmtrain(Ys, [(1:m)', K(1:m, 1:m)], ['-s 0 -c ', num2str(svmc), ' -t 4 -q 1']);
                [~, acc] = svmpredict(Yt, [(1:n)', K(m+1:end, 1:m)], model);
                fprintf('SVM = %0.4f\n', acc(1));
                
                K = TKL(Xs, Xt, options);
                model = svmtrain(Ys, [(1:m)', K(1:m, 1:m)], ['-s 0 -c ', num2str(svmc), ' -t 4 -q 1']);
                [~, acc] = svmpredict(Yt, [(1:n)', K(m+1:end, 1:m)], model);
                fprintf('TKL = %0.4f\n', acc(1));
                
                result = [result; acc(1)];
                fprintf('\n');
            end
        end
        
    case 'image'
        options.ker = 'rbf';        % kernel: 'linear' | 'rbf' | 'lap'
        options.eta = 1.1;          % eigenspectrum damping factor
        svmc = 10.0;                % SVM complexity regularizer in LibSVM
        
        srcStr = {'Caltech10', 'Caltech10', 'Caltech10', 'amazon', 'amazon', 'amazon', 'webcam', 'webcam', 'webcam', 'dslr', 'dslr', 'dslr'};
        tgtStr = {'amazon', 'webcam', 'dslr', 'Caltech10', 'webcam', 'dslr', 'Caltech10', 'amazon', 'dslr', 'Caltech10', 'amazon', 'webcam'};
        result = [];
        for iData = 1:12
            src = char(srcStr{iData});
            tgt = char(tgtStr{iData});
            data = strcat(src, '_vs_', tgt);
            
            load(['../data/' src '_SURF_L10.mat']);
            fts = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));
            Xs = zscore(fts, 1);
            Xs = Xs';
            Ys = labels;
            
            load(['../data/' tgt '_SURF_L10.mat']);
            fts = fts ./ repmat(sum(fts, 2), 1, size(fts,2));
            Xt = zscore(fts, 1);
            Xt = Xt';
            Yt = labels;
            
            fprintf('data=%s\n', data);
            Xs = bsxfun(@rdivide, Xs, sqrt(sum(Xs.^2, 1)) + eps);
            Xt = bsxfun(@rdivide, Xt, sqrt(sum(Xt.^2, 1)) + eps);
            m = size(Xs, 2);
            n = size(Xt, 2);
            
            K = kernel(options.ker, [Xs, Xt], []);
            model = svmtrain(Ys, [(1:m)', K(1:m, 1:m)], ['-s 0 -c ', num2str(svmc), ' -t 4 -q 1']);
            [~, acc] = svmpredict(Yt, [(1:n)', K(m+1:end, 1:m)], model);
            fprintf('SVM = %0.4f\n', acc(1));
            
            K = TKL(Xs, Xt, options);
            model = svmtrain(Ys, [(1:m)', K(1:m, 1:m)], ['-s 0 -c ', num2str(svmc), ' -t 4 -q 1']);
            [~, acc] = svmpredict(Yt, [(1:n)', K(m+1:end, 1:m)], model);
            fprintf('TKL = %0.4f\n', acc(1));
            
            result = [result; acc(1)];
            fprintf('\n');
        end
    otherwise
        error('domain type %s not implemented\n', domain_type);
end
