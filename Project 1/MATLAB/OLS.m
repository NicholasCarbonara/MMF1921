function  [mu, Q] = OLS(returns, factRet, lambda, K)
    
    % Use this function to perform an OLS regression. Note that you will 
    % not use lambda or K in thismodel (lambda is for LASSO, and K is for
    % BSS).
 
    % OLS regression for asset returns using factor model
   
    [T, n] = size(returns);    % T = number of months, n = number of assets
    [~, p] = size(factRet);    % p = number of factors (should be 8)
 
    X = [ones(T, 1), factRet]; % Add intercept column → T × (p+1)
    mu = zeros(n, 1);          % Initialize expected returns
    residuals = zeros(T, n);   % Store residuals to compute covariance
 
    for i = 1:n
        y = returns(:, i);                             % T × 1 vector of asset i returns
        beta = (X' * X) \ (X' * y);                    % OLS coefficients (p+1) × 1
        y_hat = X * beta;                              % Predicted returns
        residuals(:, i) = y - y_hat;                   % Store residuals
        mu(i) = mean(y_hat);                           % Expected return = mean predicted
    end
 
    Q = cov(residuals);  % n × n covariance matrix of residuals
  
    % mu =          % n x 1 vector of asset exp. returns
    % Q  =          % n x n asset covariance matrix
    %----------------------------------------------------------------------
    
end