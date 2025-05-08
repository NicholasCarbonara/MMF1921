function lambdaStar = chooseLambda(returns, factRet, lambdaGrid, k)
% Choose optimal lambda for LASSO using k-fold cross-validation
%
% returns  : T x n matrix of asset excess returns
% factRet  : T x p matrix of factor returns
% lambdaGrid : vector of candidate lambda values
% k        : number of folds for cross-validation

    T = size(returns, 1);
    idx = crossvalind('Kfold', T, k);
    cvLoss = zeros(length(lambdaGrid), 1);

    for j = 1:length(lambdaGrid)
        loss = 0;
        for fold = 1:k
            test = (idx == fold);
            train = ~test;
            [mu, ~] = LASSO(returns(train,:), factRet(train,:), lambdaGrid(j), []);
            yhat = repmat(mu', sum(test), 1);
            loss = loss + mean((returns(test,:) - yhat).^2, 'all');
        end
        cvLoss(j) = loss;
    end

    [~, bestIdx] = min(cvLoss);
    lambdaStar = lambdaGrid(bestIdx);
end