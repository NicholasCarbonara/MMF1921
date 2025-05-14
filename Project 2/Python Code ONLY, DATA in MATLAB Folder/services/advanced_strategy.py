import numpy as np
from services.advanced_estimators import AdvancedEstimators
from services.advanced_optimization import AdvancedOptimization
import pandas as pd
import cvxpy as cp

class AdvancedStrategy:
    def __init__(self, 
                 lookback_period=48,   # Longer lookback for stability
                 max_weight=0.05,     # 5% position limit
                 target_vol=0.06,     # 6% target volatility
                 rebalance_freq=18,   # Rebalance every 18 months
                 turnover_penalty=5.0):
        self.lookback_period = lookback_period
        self.max_weight = max_weight
        self.target_vol = target_vol
        self.min_vol = target_vol * 0.8
        self.rebalance_freq = rebalance_freq
        self.turnover_penalty = turnover_penalty
        self.prev_weights = None
        self.months_since_rebalance = 0

    def generate_weights(self, returns, factRet):
        if len(returns) < self.lookback_period:
            n_assets = returns.shape[1]
            return np.ones(n_assets) / n_assets
        recent_returns = returns.iloc[-self.lookback_period:]
        recent_factRet = factRet.iloc[-self.lookback_period:]
        self.months_since_rebalance += 1
        mu, Q = AdvancedEstimators.robust_return_estimation(recent_returns, recent_factRet)
        regime = self._detect_regime(recent_returns)
        risk_aversion = self._get_risk_aversion(regime)
        try:
            mv_weights = self._minimum_variance(Q)
            rp_weights = self._risk_parity(Q)
            weights = 0.7 * mv_weights + 0.3 * rp_weights
        except:
            weights = self._minimum_variance(Q)
        # --- SIMPLE POST-PROCESSING ---
        weights = np.clip(weights, 0, self.max_weight)
        weights = weights / np.sum(weights)
        if self.prev_weights is not None:
            weights = 0.6 * self.prev_weights + 0.4 * weights
        port_vol = np.sqrt(252 * weights.T @ Q @ weights)
        if port_vol > self.target_vol:
            vol_scale = self.target_vol / port_vol
            weights = weights * vol_scale
            weights = weights / np.sum(weights)
        self.prev_weights = weights.copy()
        if self.months_since_rebalance >= self.rebalance_freq:
            self.months_since_rebalance = 0
        return weights

    def _get_risk_aversion(self, regime):
        """
        Dynamic risk aversion based on volatility regime
        """
        if regime == 'low_vol_up':
            return 0.5  # More aggressive in favorable conditions
        elif regime == 'high_vol_up':
            return 0.8  # Still fairly aggressive
        elif regime == 'low_vol_down':
            return 1.2  # Moderately conservative
        else:  # high_vol_down
            return 1.5  # Most conservative

    def _minimum_variance(self, Q):
        """
        Minimum variance portfolio optimization
        """
        n = Q.shape[0]
        w = cp.Variable(n)
        
        # Objective: minimize portfolio variance
        objective = cp.Minimize(cp.quad_form(w, Q))
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Full investment
            w >= 0,          # Long only
            w <= self.max_weight  # Position limits
        ]
        
        # Solve optimization
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        
        if w.value is not None:
            weights = np.array(w.value)
            weights = np.maximum(weights, 0)  # Ensure non-negative
            weights = weights / np.sum(weights)  # Normalize
            return weights
        else:
            return np.ones(n) / n  # Equal weight if optimization fails

    def _risk_parity(self, Q):
        """
        Risk parity portfolio optimization
        """
        n = Q.shape[0]
        w = cp.Variable(n)
        
        # Calculate risk contributions
        risk_contrib = cp.multiply(w, Q @ w)
        
        # Objective: minimize sum of squared differences between risk contributions
        objective = cp.Minimize(cp.sum_squares(risk_contrib - cp.sum(risk_contrib)/n))
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Full investment
            w >= 0,          # Long only
            w <= self.max_weight  # Position limits
        ]
        
        # Solve optimization
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        
        if w.value is not None:
            weights = np.array(w.value)
            weights = np.maximum(weights, 0)  # Ensure non-negative
            weights = weights / np.sum(weights)  # Normalize
            return weights
        else:
            return np.ones(n) / n  # Equal weight if optimization fails

    def _detect_regime(self, returns):
        """
        Detect market regime based on returns and volatility
        """
        # Calculate rolling statistics
        vol = returns.std().mean() * np.sqrt(252)  # Average volatility across assets
        ret = returns.mean().mean() * 252  # Average return across assets
        
        # Determine regime
        if vol < returns.std().mean() * np.sqrt(252):  # Compare to average volatility
            if ret > 0:
                return 'low_vol_up'
            else:
                return 'low_vol_down'
        else:
            if ret > 0:
                return 'high_vol_up'
            else:
                return 'high_vol_down'