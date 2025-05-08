function Kstar = chooseK(returns, factRet, Kgrid, k)
% Choose optimal number of factors K for BSS using k-fold cross-validation
%
% returns  : T x n matrix of asset excess returns
% factRet  : T x p matrix of factor returns
% Kgrid    : vector of candidate K values
% k        : number of folds

    T = size(returns, 1);
    idx = crossvalind('Kfold', T, k);
    cvLoss = zeros(length(Kgrid), 1);

    for j = 1:length(Kgrid)
        loss = 0;
        for fold = 1:k
            test = (idx == fold);
            train = ~test;
            [mu, ~] = BSS(returns(train,:), factRet(train,:), [], Kgrid(j));
            yhat = repmat(mu', sum(test), 1);
            loss = loss + mean((returns(test,:) - yhat).^2, 'all');
        end
        cvLoss(j) = loss;
    end

    [~, bestIdx] = min(cvLoss);
    Kstar = Kgrid(bestIdx);
end