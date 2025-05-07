function [mu, Q] = LASSO(returns, factRet, lambda, ~)
    %-----------------------------------------------------------------------
    % LASSO FACTOR MODEL
    % returns : m×n matrix of asset *excess* returns (m obs, n assets)
    % factRet : m×p matrix of factor returns (same m obs, p=8)
    % lambda  : ℓ₁ penalty parameter (choose in main script)
    % ~       : dummy input – K is unused for LASSO
    %
    % mu : n×1 vector of expected excess returns
    % Q  : n×n covariance matrix of excess returns
    %
    % Method:
    % For each asset i we solve
    %   min_{β_i} ‖y_i–Xβ_i‖² + λ‖β_i‖₁,
    % where X = [1 F] contains an intercept and all eight factors.
    % The ℓ₁-norm is implemented with the usual (β⁺, β⁻) split so that the
    % problem becomes a convex Quadratic Program:
    %
    %   min_{z≥0} ½ zᵀH z + fᵀz,
    % which `quadprog` can solve efficiently.
    %
    % Once the coefficients are found, we build μ and Q using:
    %   μ = α + B · μ_f,
    %   Q = B Σ_f Bᵀ + diag(σ²),
    % where B = factor loadings (no intercept),
    % Σ_f = factor-return covariance,
    % σ² = residual variances.
    %-----------------------------------------------------------------------

    % Dimensions
    [m, n] = size(returns);     % obs × assets
    p = size(factRet, 2);       % # factors = 8
    d = p + 1;                  % +1 for the intercept

    % Design matrix with intercept
    X = [ones(m, 1), factRet];  % m × d

    % Pre-allocate
    alpha = zeros(n, 1);        % intercepts
    B = zeros(n, p);            % factor loadings
    sigma2 = zeros(n, 1);       % residual variances

    % ----- build constant pieces used in every loop ------------------------
    Xt = X';                                   % cache transpose
    mu_f = mean(factRet, 1)';                  % p×1, factor means
    Sigma_f = cov(factRet, 1);                 % p×p, uses 1/m normalization

    % Quadprog options (silent)
    opts = optimoptions('quadprog', 'Display', 'off');

    % LOOP THROUGH ASSETS ---------------------------------------------------
    for i = 1:n
        y = returns(:, i);                     % m×1

        % Build the (β⁺, β⁻) split ------------------------------------------
        % β = β⁺ – β⁻ , β⁺, β⁻ ≥ 0
        % z = [β⁺ ; β⁻] ∈ ℝ^{2d}
        A_split = [eye(d), -eye(d)];          % d × 2d
        A = X * A_split;                      % m × 2d

        % Quadratic and linear terms for ½ zᵀHz + fᵀz
        H = (2 / m) * (A' * A);              % 2d × 2d, PSD
        f = lambda * ones(2 * d, 1) - (2 / m) * (A' * y);

        % Non-negativity bounds
        lb = zeros(2 * d, 1);
        ub = [];
        Aineq = [];
        bineq = [];
        Aeq = [];
        beq = [];

        % Solve the QP
        z = quadprog(H, f, Aineq, bineq, Aeq, beq, lb, ub, [], opts);

        % Recover β ---------------------------------------------------------
        beta = z(1:d) - z(d + 1:end);        % d×1
        alpha(i) = beta(1);                  % scalar intercept
        B(i, :) = beta(2:end)';              % 1×p

        resid = y - X * beta;                % m×1
        sigma2(i) = var(resid, 1);           % population residual variance
    end

    % EXPECTED RETURNS ------------------------------------------------------
    mu = alpha + B * mu_f;                   % n×1

    % COVARIANCE MATRIX -----------------------------------------------------
    Q = B * Sigma_f * B' + diag(sigma2);     % n×n
end
