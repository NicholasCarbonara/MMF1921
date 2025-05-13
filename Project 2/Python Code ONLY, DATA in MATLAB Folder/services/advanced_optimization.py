import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

class AdvancedOptimization:
    @staticmethod
    def risk_parity(mu, Q, max_weight=0.1):
        """
        Implements Risk Parity optimization
        """
        n = len(mu)
        
        def risk_contribution(w, Q):
            portfolio_risk = np.sqrt(w.T @ Q @ w)
            marginal_risk = Q @ w
            risk_contribution = w * marginal_risk / portfolio_risk
            return risk_contribution
        
        def objective(w):
            risk_contrib = risk_contribution(w, Q)
            target_risk = np.ones(n) / n
            return np.sum((risk_contrib - target_risk)**2)
        
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
    def enhanced_mvo(mu, Q, target_vol=None, max_weight=0.1):
        """
        Enhanced MVO with volatility targeting and position limits
        """
        n = len(mu)
        
        # Define variables
        x = cp.Variable(n)
        
        # Objective: maximize expected return
        objective = cp.Maximize(mu.T @ x)
        
        # Constraints
        constraints = [
            cp.sum(x) == 1,  # weights sum to 1
            x >= 0,  # no short selling
            x <= max_weight,  # maximum position size
        ]
        
        # Add volatility constraint if target_vol is provided
        if target_vol is not None:
            constraints.append(cp.quad_form(x, Q) <= target_vol**2)
        
        # Solve
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=False)
        
        return x.value

    @staticmethod
    def min_variance(Q, max_weight=0.1):
        """
        Minimum Variance optimization
        """
        n = Q.shape[0]
        
        # Define variables
        x = cp.Variable(n)
        
        # Objective: minimize variance
        objective = cp.Minimize(cp.quad_form(x, Q))
        
        # Constraints
        constraints = [
            cp.sum(x) == 1,  # weights sum to 1
            x >= 0,  # no short selling
            x <= max_weight,  # maximum position size
        ]
        
        # Solve
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=False)
        
        return x.value

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