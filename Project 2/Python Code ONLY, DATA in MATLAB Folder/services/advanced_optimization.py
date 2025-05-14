import cvxpy as cp
import numpy as np
from scipy.optimize import minimize
from scipy import linalg

class AdvancedOptimization:
    @staticmethod
    def risk_parity(mu, Q, max_weight=0.05, prev_weights=None, turnover_penalty=0.1):
        """
        Enhanced Risk Parity with turnover control and return consideration
        """
        n = len(mu)
        
        def risk_contribution(w):
            w = w.reshape(-1)
            portfolio_risk = np.sqrt(w.T @ Q @ w)
            risk_contrib = w * (Q @ w) / portfolio_risk
            return risk_contrib
        
        def objective(w):
            w = w.reshape(-1)
            risk_contrib = risk_contribution(w)
            target_risk = np.ones(n) / n
            
            # Add return consideration
            ret = mu.T @ w
            risk = np.sqrt(w.T @ Q @ w)
            sharpe_penalty = -ret / (risk + 1e-8)  # Negative because we're minimizing
            
            # Add turnover penalty if previous weights exist
            if prev_weights is not None:
                turnover = np.sum(np.abs(w - prev_weights))
                return np.sum((risk_contrib - target_risk)**2) + 0.5 * sharpe_penalty + turnover_penalty * turnover
            
            return np.sum((risk_contrib - target_risk)**2) + 0.5 * sharpe_penalty
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        if prev_weights is not None:
            constraints.append(
                {'type': 'ineq', 'fun': lambda x: 0.4 - np.sum(np.abs(x - prev_weights))}  # turnover <= 40%
            )
        
        bounds = [(0, max_weight) for _ in range(n)]
        
        # Multi-start optimization
        best_w = None
        best_obj = np.inf
        
        # Try different starting points
        starts = [
            np.ones(n) / n,  # Equal weight
            np.minimum(np.maximum(mu.flatten() / np.sum(np.abs(mu)), 0), max_weight),  # Return-based
            np.minimum(1/np.sqrt(np.diag(Q)) / np.sum(1/np.sqrt(np.diag(Q))), max_weight)  # Vol-based
        ]
        
        if prev_weights is not None:
            starts.append(prev_weights)  # Previous weights
        
        for w0 in starts:
            result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success and result.fun < best_obj:
                best_obj = result.fun
                best_w = result.x
        
        if best_w is None:
            return np.ones(n) / n
            
        return best_w

    @staticmethod
    def max_diversification(mu, Q, max_weight=0.1):
        """
        Implements Maximum Diversification optimization
        """
        n = len(mu)
        
        def diversification_ratio(w, Q):
            portfolio_risk = np.sqrt(w.T @ Q @ w)
            weighted_vol = np.sum(w * np.sqrt(np.diag(Q)))
            return weighted_vol / portfolio_risk
        
        def objective(w):
            return -diversification_ratio(w, Q)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        ]
        bounds = [(0, max_weight) for _ in range(n)]  # no short selling and max weight
        
        # Initial guess
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x

    @staticmethod
    def max_sharpe_ratio(mu, Q, rf_rate=0.02, max_weight=0.05, prev_weights=None, turnover_penalty=10.0):
        """
        Maximum Sharpe Ratio optimization with enhanced risk-return balance
        """
        n = len(mu)
        w = cp.Variable(n)
        
        # Objective terms
        ret = mu.T @ w - rf_rate
        risk = cp.sqrt(cp.quad_form(w, Q))
        sharpe = ret / risk
        
        # Enhanced regularization
        l2_reg = 0.1 * cp.sum_squares(w)  # L2 regularization
        entropy_reg = -0.1 * cp.sum(cp.entr(w + 1e-8))  # Entropy regularization for diversification
        
        # Objective with regularization
        if prev_weights is not None:
            turnover = cp.sum(cp.abs(w - prev_weights))
            objective = sharpe - turnover_penalty * turnover - l2_reg - entropy_reg
        else:
            objective = sharpe - l2_reg - entropy_reg
        
        # Enhanced constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= max_weight,
            risk <= 0.12,  # 12% volatility constraint
            ret >= 0.085,  # Minimum return constraint
        ]
        
        if prev_weights is not None:
            constraints.append(cp.sum(cp.abs(w - prev_weights)) <= 0.05)  # 5% turnover constraint
        
        # Multi-start optimization with different settings
        best_w = None
        best_sharpe = -np.inf
        
        # Different solver configurations
        solver_configs = [
            {'max_iters': 2000, 'abstol': 1e-7, 'reltol': 1e-6, 'feastol': 1e-7},
            {'max_iters': 1000, 'abstol': 1e-6, 'reltol': 1e-5, 'feastol': 1e-6},
            {'max_iters': 3000, 'abstol': 1e-8, 'reltol': 1e-7, 'feastol': 1e-8}
        ]
        
        # Different initial points
        initial_points = [
            np.ones(n) / n,  # Equal weight
            np.minimum(np.maximum(mu.flatten() / np.sum(np.abs(mu)), 0), max_weight),  # Return-based
            np.minimum(1/np.sqrt(np.diag(Q)) / np.sum(1/np.sqrt(np.diag(Q))), max_weight),  # Risk-based
            prev_weights if prev_weights is not None else np.ones(n) / n  # Previous weights
        ]
        
        for config in solver_configs:
            for init_w in initial_points:
                try:
                    # Warm start
                    w.value = init_w
                    
                    prob = cp.Problem(cp.Maximize(objective), constraints)
                    prob.solve(solver=cp.ECOS, warm_start=True, **config)
                    
                    if prob.status == 'optimal':
                        curr_w = np.array(w.value)
                        curr_ret = float(mu.T @ curr_w)
                        curr_risk = float(np.sqrt(curr_w.T @ Q @ curr_w))
                        curr_sharpe = (curr_ret - rf_rate) / curr_risk
                        
                        if curr_sharpe > best_sharpe:
                            best_sharpe = curr_sharpe
                            best_w = curr_w
                except:
                    continue
        
        if best_w is not None:
            return best_w
        
        # If optimization fails, try risk parity approach
        try:
            return AdvancedOptimization.risk_parity(mu, Q, max_weight, prev_weights, turnover_penalty)
        except:
            return AdvancedOptimization.min_variance(Q, max_weight)

    @staticmethod
    def min_variance(Q, max_weight=0.05):
        """
        Minimum variance optimization with position limits
        """
        n = Q.shape[0]
        w = cp.Variable(n)
        
        risk = cp.quad_form(w, Q)
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= max_weight
        ]
        
        prob = cp.Problem(cp.Minimize(risk), constraints)
        try:
            prob.solve(solver=cp.ECOS)
            if prob.status == 'optimal':
                return np.array(w.value)
            else:
                return np.ones(n) / n  # Equal weight fallback
        except:
            return np.ones(n) / n  # Equal weight fallback

    @staticmethod
    def mean_variance(mu, Q, risk_aversion=2.0, max_weight=0.05, prev_weights=None, turnover_penalty=10.0):
        """
        Mean-variance optimization with enhanced stability and strict turnover control
        """
        n = len(mu)
        w = cp.Variable(n)
        
        # Objective terms
        ret = mu.T @ w
        risk = cp.quad_form(w, Q)
        objective = ret - risk_aversion * risk
        
        # Add turnover penalty
        if prev_weights is not None:
            turnover = cp.sum(cp.abs(w - prev_weights))
            objective -= turnover_penalty * turnover
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= max_weight,
            cp.sqrt(risk) <= 0.12,  # 12% volatility constraint
            cp.sum(cp.abs(w - prev_weights)) <= 0.05 if prev_weights is not None else True  # Explicit 5% turnover constraint
        ]
        
        # Solve optimization problem
        prob = cp.Problem(cp.Maximize(objective), constraints)
        try:
            prob.solve(solver=cp.ECOS, max_iters=1000)
            if prob.status == 'optimal':
                return np.array(w.value)
            else:
                return AdvancedOptimization.min_variance(Q, max_weight)
        except:
            return AdvancedOptimization.min_variance(Q, max_weight)

def risk_parity_weights(cov_matrix, max_weight=0.1):
    """
    Compute risk parity weights using scipy's minimize
    """
    n = cov_matrix.shape[0]
    
    def risk_contribution(w):
        portfolio_risk = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        risk_contrib = np.multiply(np.dot(cov_matrix, w), w) / portfolio_risk
        return risk_contrib
    
    def objective(w):
        risk_contrib = risk_contribution(w)
        target_risk = np.ones(n) / n  # Equal risk contribution
        return np.sum((risk_contrib - target_risk)**2)
    
    # Initial guess (equal weights)
    w0 = np.ones(n) / n
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
    ]
    bounds = [(0, max_weight) for _ in range(n)]  # bounds for each weight
    
    # Optimize
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
        # Fallback to equal weight if optimization fails
        return np.ones(n) / n
    
    return result.x 