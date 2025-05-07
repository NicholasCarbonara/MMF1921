function x = MVO(mu, Q, targetRet)
%MVO  Mean-Variance Optimizer (long-only, fully-invested)
%
%   x = MVO(mu, Q, targetRet) returns the optimal portfolio weights that
%   minimize portfolio variance subject to:
%       • required expected excess return ≥ targetRet
%       • no short-selling (x ≥ 0)
%       • full investment (∑ x = 1)
%
%   Inputs:
%       mu        : n-by-1 vector of expected excess returns
%       Q         : n-by-n covariance matrix of asset returns
%       targetRet : scalar target excess return
%
%   Output:
%       x         : n-by-1 vector of optimal weights
%
%   The problem solved is:
%
%         minimize      xᵀ Q x
%         subject to    μᵀ x  ≥  targetRet
%                       1ᵀ x  =  1
%                       x      ≥  0
%
%   This is a convex quadratic programme, solved below with MATLAB’s
%   QUADPROG. If QUADPROG fails (e.g., because the target return is
%   infeasible), the code relaxes the target-return constraint gracefully by
%   re-optimizing without it and issues a warning.
%
%   ---------------------------------------------------------------------

    % ----- 1. Dimensions ----------------------------
    n = length(mu);                 % Number of assets

    % ----- 2. Quadratic objective -------------------
    % quadprog solves   ½ xᵀ H x + fᵀ x
    % so pass H = 2Q to match our objective xᵀ Q x
    H = 2 * Q;
    f = zeros(n, 1);

    % ----- 3. Linear constraints --------------------
    % (a) Expected return  μᵀ x ≥ targetRet   ⇔  −μᵀ x ≤ −targetRet
    A  = -mu(:)';                  % 1-by-n
    b  = -targetRet;

    % (b) Fully invested  1ᵀ x = 1
    Aeq = ones(1, n);
    beq = 1;

    % (c) Long-only bounds
    lb = zeros(n, 1);              % x ≥ 0
    ub = [];                      % no upper bound needed (weights ≤ 1 is implied by ∑x=1 & x≥0)

    % ----- 4. Solve with quadprog -------------------
    opts = optimoptions('quadprog', ...
                        'Display', 'off', ...
                        'Algorithm', 'interior-point-convex');

    [x, ~, exitflag] = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], opts);

    % ----- 5. Feasibility check & graceful fallback --
    if exitflag ~= 1
        warning('MVO:InfeasibleTarget', ...
                'Target return %.4f infeasible. Re-optimizing without the return constraint.', ...
                targetRet);

        % Remove the return constraint and minimize variance subject to
        % long-only & fully-invested
        [x, ~, exitflag] = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], opts);

        if exitflag ~= 1
            error('MVO:OptimizationFailed', 'quadprog failed to converge.');
        end
    end
end
