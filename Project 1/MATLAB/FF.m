function [mu, Q] = FF(returns, factRet, ~, ~)
% FF  –  Fama‑French 3‑factor model (no regularisation, no toolbox needed)
%
%   [mu, Q] = FF(returns, factRet, ~, ~)
%
% INPUTS
%   returns : (T × n) matrix of asset EXCESS returns  r_i - r_f
%   factRet : (T × p) matrix of factor EXCESS returns (first three columns
%             must be Mkt‑RF, SMB, HML; any extra columns are ignored)
%   ~       : placeholders for lambda and K (unused here but kept so the
%             function matches the common interface expected by FMList)
%
% OUTPUTS
%   mu  : (n × 1) vector of expected excess returns of each asset
%   Q   : (n × n) covariance matrix  B Σ_f Bᵀ + diag(σ²_ε)
%
% METHOD
%   Ordinary least‑squares regression with an intercept on the three FF
%   factors for every asset.  Closed‑form solution via X\Y (no toolboxes).

% -------------------------------------------------------------------------
% 1. Data dimensions & factor subset
% -------------------------------------------------------------------------
[T, n] = size(returns);

% Keep *only* the three Fama‑French factors (assumed to be columns 1‑3)
F      = factRet(:, 1:3);          % (T × k) with k = 3
k      = 3;

% -------------------------------------------------------------------------
% 2. OLS coefficients (α and β) for *all* assets at once
% -------------------------------------------------------------------------
X      = [ones(T,1)  F];           % (T × (k+1))  – intercept + factors
coef   = X \ returns;              % (k+1 × n)    – OLS via backslash

alpha  = coef(1, :)';              % (n × 1) intercepts
B      = coef(2:end, :)';          % (n × k) factor loadings

% -------------------------------------------------------------------------
% 3. Factor and idiosyncratic moments
% -------------------------------------------------------------------------
fBar   = mean(F, 1)';              % E[f]          (k × 1)
Sigma_f = cov(F, 1);               % Σ_f (ML)      (k × k)

resid  = returns - X * coef;       % (T × n) residuals
sigma2 = var(resid, 1);            % (1 × n)  ML idiosyncratic variances

% -------------------------------------------------------------------------
% 4. Outputs  μ and Q
% -------------------------------------------------------------------------
mu = alpha + B * fBar;                     % (n × 1)

Q  = B * Sigma_f * B.' + diag(sigma2);     % (n × n)
end
