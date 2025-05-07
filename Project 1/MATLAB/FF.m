function [mu, Q] = LASSO(returns, factRet, lambda, ~)
% LASSO  –  Fama‑French 3‑factor model with ℓ1 regularisation
%
%   [mu, Q] = LASSO(returns, factRet, lambda, ~)
%
% INPUTS
%   returns  :  (T × n)  matrix of asset returns in EXCESS of r_f
%   factRet  :  (T × k)  matrix of factor returns in EXCESS of r_f (k = 3)
%   lambda   :  scalar   ℓ1‑penalty weight (same λ for every asset)
%   ~        :  (ignored)  placeholder for K (used only in BSS task)
%
% OUTPUTS
%   mu   :  (n × 1)  expected excess return of each asset
%   Q    :  (n × n)  covariance matrix  B Σ_f B' + diag(σ^2_ε)
%
% METHOD
%   For each asset i we solve
%
%     min_{α_i,β_i}  ½‖r_i − α_i·1 − F β_i‖_2²  + λ‖β_i‖_1
%
%   via quadratic programming after splitting  β_i = β⁺ − β⁻ ,
%   β⁺,β⁻ ≥ 0.  No high‑level regression helpers are used.

% -------------------------------------------------------------------------
% 1.  House‑keeping
% -------------------------------------------------------------------------
[T, n] = size(returns);
k      = size(factRet, 2);           % = 3 for FF‑3
F      = factRet;                    % alias (T × k)
oneT   = ones(T,1);                  % (T × 1)

% containers
alpha  = zeros(n,1);
B      = zeros(n,k);
sigma2 = zeros(n,1);

% Pre‑compute factor moments (needed later for mu, Q) ---------------------
f_bar   = mean(F,        1 ).';      % (k × 1)   E[f]
Sigma_f = cov (F, 'omitrows');       % (k × k)   Σ_f

% -------------------------------------------------------------------------
% 2.  Build quadratic‑program matrices common to EVERY asset
% -------------------------------------------------------------------------
% Decision variables for each asset:
%   z = [ α ;  β⁺ ; β⁻ ]    length = 1 + k + k
% Predictor matrix that maps z → α·1 + F(β⁺−β⁻)
A = [oneT ,  F , -F];                % (T × (1+2k))
H_common = A.' * A;                  % part of quadratic term
% make H strictly positive‑definite for numerical stability
epsPD = 1e-10;
H_common = H_common + epsPD * eye(1+2*k);

% Linear penalty part for λ‖β‖₁  =>  λ·1ᵀ(β⁺+β⁻)
c_pen = [0 ; ones(2*k,1)];           % selects only β⁺,β⁻

optsQP = optimoptions('quadprog','Display','off');

% -------------------------------------------------------------------------
% 3.  Loop through assets – solve LASSO QP for each
% -------------------------------------------------------------------------
for i = 1:n
    y = returns(:,i);                % (T × 1)
    f_lin = -A.' * y  +  lambda * c_pen;   % linear term of QP
    
    % lower bounds: α free, β⁺, β⁻ ≥ 0
    lb = [-inf ; zeros(2*k,1)];
    
    % solve:   min ½ zᵀ H z  + fᵀ z
    z = quadprog( H_common, f_lin, [],[], [],[], lb, [], [], optsQP );
    
    % unpack solution -----------------------------------------------------
    alpha(i)  = z(1);
    beta_pos  = z(2           :1+k);
    beta_neg  = z(2+k+1 :end );
    B(i,:)    = (beta_pos - beta_neg).';      % (1 × k)
    
    % residuals & idiosyncratic variance
    resid     = y - A*z;
    sigma2(i) = var(resid, 1);      % ML estimator (divide by T)
end

% -------------------------------------------------------------------------
% 4.  Package outputs  (μ  &  Q)
% -------------------------------------------------------------------------
mu = alpha + B * f_bar;                     % (n × 1)

Q  = B * Sigma_f * B.' + diag(sigma2);      % (n × n)
end
