function  [mu, Q] = BSS(returns, factRet, ~, K)
% -------------------------------------------------------------------------
% Best Subset Selection (BSS) Factor Model
%
% INPUTS
%   returns : T×n matrix of asset excess returns      (ri – rf)
%   factRet : T×p matrix of candidate factor returns (fm – rf, SMB, HML, …)
%   ~       : placeholder for lambda (unused here)
%   K       : number of factors to keep (cardinality of β)
%
% OUTPUTS
%   mu  : n×1 expected excess return vector
%   Q   : n×n factor‑based covariance matrix
%
% METHOD
%   • For each asset i, search all (p choose K) subsets of factors.
%   • Run OLS with an intercept on each subset, keep the one with
%     minimum residual sum of squares (RSS).  This exactly solves
%     the BSS problem when p is modest (≤ ~15–20); for larger p
%     swap the inner loop for a MIQP solver such as GUROBI or CPLEX
%     using the MIO formulation in §2.2 of Bertsimas et al. (2016).
% -------------------------------------------------------------------------
[T,n] = size(returns);
p     = size(factRet,2);

% --- pre‑compute factor moments ------------------------------------------
fBar     = mean(factRet,1)';                 % E[f]   (p×1)
Sigma_f  = cov(factRet,1);                   % Σ_f    (p×p)   population estimator

% --- all K‑subsets of {1,…,p} --------------------------------------------
combIdx  = nchoosek(1:p, K);                 % (#comb × K)
nComb    = size(combIdx,1);

% --- storage --------------------------------------------------------------
B          = zeros(n,p);                     % factor loadings
alpha      = zeros(n,1);                     % intercepts
sigma2_eps = zeros(n,1);                     % idiosyncratic variances

% --- loop over assets -----------------------------------------------------
for i = 1:n
    y        = returns(:,i);                 % T×1
    bestRSS  = inf;
    bestSet  = [];
    bestCoef = [];
    
    % Search over all subsets
    for c = 1:nComb
        S         = combIdx(c,:);
        X         = [ones(T,1)  factRet(:,S)];   % intercept + chosen factors
        coef      = X \ y;                      % OLS (K+1)×1
        resid     = y - X*coef;
        RSS       = resid' * resid;
        
        if RSS < bestRSS
            bestRSS  = RSS;
            bestSet  = S;
            bestCoef = coef;
        end
    end
    
    % store coefficients
    alpha(i)          = bestCoef(1);          % intercept
    b_i               = zeros(1,p);
    b_i(bestSet)      = bestCoef(2:end)';     % chosen betas
    B(i,:)            = b_i;
    sigma2_eps(i)     = bestRSS / (T - (K+1));
end

% --- model‑implied moments -----------------------------------------------
mu = alpha + B * fBar;                        % expected return (n×1)
Q  = B * Sigma_f * B' + diag(sigma2_eps);     % covariance     (n×n)
end