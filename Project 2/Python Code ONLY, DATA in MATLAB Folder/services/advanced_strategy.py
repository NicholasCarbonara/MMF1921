import numpy as np
from services.advanced_estimators import AdvancedEstimators
from services.advanced_optimization import AdvancedOptimization
import pandas as pd
import cvxpy as cp

class AdvancedStrategy:
    def __init__(self, 
                 lookback_period=36,
                 max_weight=0.05,      # 5% position limit
                 target_vol=0.12,      # 12% target volatility
                 rebalance_freq=12,    # Annual rebalancing
                 turnover_penalty=10.0):
        self.lookback_period = lookback_period
        self.max_weight = max_weight
        self.target_vol = target_vol
        self.min_vol = target_vol * 0.8  # Allow minimum volatility of 80% of target
        self.rebalance_freq = rebalance_freq
        self.turnover_penalty = turnover_penalty
        self.prev_weights = None
        self.months_since_rebalance = 0
        
    def generate_weights(self, returns, factRet):
        """
        Generate portfolio weights with focus on Sharpe ratio and low turnover
        """
        if len(returns) < self.lookback_period:
            n_assets = returns.shape[1]
            return np.ones(n_assets) / n_assets
        
        # Use recent data for estimation
        recent_returns = returns.iloc[-self.lookback_period:]
        recent_factRet = factRet.iloc[-self.lookback_period:]
        
        # Increment months since last rebalance
        self.months_since_rebalance += 1
        
        # Check current portfolio volatility
        if self.prev_weights is not None:
            port_vol = np.sqrt(252 * self.prev_weights.T @ np.cov(recent_returns.T) @ self.prev_weights)
            
            # More aggressive volatility targeting
            if port_vol < self.min_vol:  # If below minimum volatility, increase risk
                vol_scale = self.target_vol / port_vol
                scaled_weights = self.prev_weights * vol_scale
                if not self.months_since_rebalance >= self.rebalance_freq:
                    return scaled_weights / np.sum(scaled_weights)
        
        # Enhanced parameter estimation
        mu, Q = AdvancedEstimators.robust_return_estimation(recent_returns, recent_factRet)
        
        # Get current regime and risk parameters
        regime = self._detect_regime(recent_returns)
        risk_aversion = self._get_risk_aversion(regime)
        
        # Dynamic turnover adjustment - more lenient in favorable conditions
        if regime in ['low_vol_up', 'high_vol_up']:
            turnover_mult = 0.8  # More trading allowed in favorable conditions
        else:
            turnover_mult = 1.5  # Still conservative in unfavorable conditions
        
        # Optimize portfolio
        try:
            weights = self._optimize(
                mu, Q,
                prev_weights=self.prev_weights,
                turnover_mult=turnover_mult,
                risk_aversion=risk_aversion
            )
        except:
            weights = self._minimum_variance(Q)
        
        # Apply position limits
        weights = np.minimum(weights, self.max_weight)
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
        # More aggressive volatility targeting
        port_vol = np.sqrt(252 * weights.T @ Q @ weights)
        if port_vol < self.target_vol * 0.9:  # Scale up if significantly below target
            vol_scale = self.target_vol / port_vol
            weights = weights * vol_scale
            weights = weights / np.sum(weights)
        
        # Update state
        self.prev_weights = weights.copy()
        if self.months_since_rebalance >= self.rebalance_freq:
            self.months_since_rebalance = 0
        
        return weights

    def _get_risk_aversion(self, regime):
        """
        Dynamic risk aversion based on volatility regime - more aggressive settings
        """
        if regime == 'low_vol_up':
            return 0.5  # More aggressive in favorable conditions
        elif regime == 'high_vol_up':
            return 0.8  # Still fairly aggressive in good conditions
        elif regime == 'low_vol_down':
            return 1.2  # Moderately conservative
        else:  # high_vol_down
            return 1.5  # Less conservative than before

    def optimize_portfolio(self, returns, factors=None, prev_weights=None):
        """
        Enhanced portfolio optimization with regime-based adjustments
        """
        regime = self._detect_regime(returns)
        
        # Adjust parameters based on regime
        if regime == 'low_vol_up':
            turnover_mult = 0.5  # Allow more turnover in favorable conditions
            risk_aversion = 0.8  # More aggressive
        elif regime == 'high_vol_up':
            turnover_mult = 1.0
            risk_aversion = 1.2  # More conservative
        elif regime == 'low_vol_down':
            turnover_mult = 2.0  # Reduce turnover in unfavorable conditions
            risk_aversion = 1.5  # More conservative
        else:  # high_vol_down
            turnover_mult = 3.0  # Minimize turnover in highly unfavorable conditions
            risk_aversion = 2.0  # Most conservative
            
        # Calculate expected returns and covariance
        mu = self._estimate_returns(returns, factors)
        Q = self._estimate_covariance(returns)
        
        # Optimize portfolio
        try:
            weights = self._optimize(mu, Q, prev_weights, 
                                   turnover_mult=turnover_mult,
                                   risk_aversion=risk_aversion)
        except:
            # Fallback to minimum variance if optimization fails
            weights = self._minimum_variance(Q)
            
        return weights
        
    def _optimize(self, mu, Q, prev_weights=None, turnover_mult=1.0, risk_aversion=1.0):
        """
        Enhanced portfolio optimization with multiple risk constraints
        """
        n = len(mu)
        w = cp.Variable(n)
        
        # Expected return and risk
        port_return = mu.T @ w
        port_risk = cp.quad_form(w, Q)
        
        # Base objective with dynamic risk adjustment
        objective = port_return - risk_aversion * port_risk
        
        # Turnover penalty if previous weights exist
        if prev_weights is not None:
            turnover = cp.sum(cp.abs(w - prev_weights))
            objective -= self.turnover_penalty * turnover_mult * turnover
        
        # Basic constraints
        constraints = [
            cp.sum(w) == 1,  # Full investment
            w >= 0,          # Long only
            w <= self.max_weight  # Position limits
        ]
        
        # Risk factor constraints
        if prev_weights is not None:
            # Limit tracking error to previous portfolio
            tracking_error = cp.quad_form(w - prev_weights, Q)
            constraints.append(tracking_error <= 0.02)  # 2% tracking error limit
        
        # Sector/factor exposure constraints
        eigenvals, eigenvecs = np.linalg.eigh(Q)
        top_k_factors = 3  # Consider top 3 risk factors
        for k in range(top_k_factors):
            factor_exposure = eigenvecs[:, -(k+1)].T @ w
            constraints.append(factor_exposure >= -0.3)  # Lower bound
            constraints.append(factor_exposure <= 0.3)   # Upper bound
        
        # Dynamic risk constraints based on regime
        vol_regime = self._detect_regime(mu)
        if vol_regime in ['high_vol_up', 'high_vol_down']:
            # Add concentration limits in high volatility regimes
            effective_n = int(1/self.max_weight)  # Effective number of positions
            sum_sq_weights = cp.sum_squares(w)
            constraints.append(sum_sq_weights <= 1/effective_n)
        
        # Solve optimization problem with multiple attempts
        for solver in [cp.ECOS, cp.SCS]:
            try:
                prob = cp.Problem(cp.Maximize(objective), constraints)
                prob.solve(solver=solver, max_iters=1000)
                
                if w.value is not None:
                    weights = np.array(w.value)
                    weights = np.maximum(weights, 0)  # Ensure non-negative
                    weights = weights / np.sum(weights)  # Normalize
                    return weights
            except:
                continue
        
        # Fallback to minimum variance if optimization fails
        return self._minimum_variance(Q)
        
    def _minimum_variance(self, Q):
        """
        Enhanced minimum variance portfolio with stability improvements
        """
        n = Q.shape[0]
        w = cp.Variable(n)
        
        # Objective: minimize portfolio variance
        objective = cp.quad_form(w, Q)
        
        # Basic constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= self.max_weight
        ]
        
        # Add minimum effective positions constraint
        sum_sq_weights = cp.sum_squares(w)
        min_effective_positions = n/4  # At least 1/4 of available assets
        constraints.append(1/sum_sq_weights >= min_effective_positions)
        
        # Solve with multiple attempts
        for solver in [cp.ECOS, cp.SCS]:
            try:
                prob = cp.Problem(cp.Minimize(objective), constraints)
                prob.solve(solver=solver)
                
                if w.value is not None:
                    weights = np.array(w.value)
                    weights = np.maximum(weights, 0)
                    return weights / np.sum(weights)
            except:
                continue
        
        # Ultimate fallback: equal weight
        return np.ones(n) / n

    def _detect_regime(self, returns):
        """
        Enhanced regime detection using multiple indicators
        """
        if isinstance(returns, pd.Series):
            returns = returns.to_frame()
            
        # Convert to numpy array for calculations
        returns_np = returns.values
        
        # Calculate multiple regime indicators
        # 1. Volatility regime
        rolling_vol = np.std(returns_np[-12:]) * np.sqrt(252)  # 1-year rolling vol
        long_vol = np.std(returns_np) * np.sqrt(252)  # Full period vol
        vol_ratio = rolling_vol / long_vol
        
        # 2. Return regime
        rolling_ret = np.mean(returns_np[-12:]) * 252  # 1-year rolling return
        long_ret = np.mean(returns_np) * 252  # Full period return
        ret_ratio = rolling_ret / (long_ret + 1e-8)  # Avoid division by zero
        
        # 3. Drawdown regime
        cumret = np.cumprod(1 + returns_np.flatten())
        rolling_max = np.maximum.accumulate(cumret)
        drawdown = (cumret - rolling_max) / rolling_max
        current_dd = drawdown[-1]
        
        # Combine indicators for regime classification
        vol_score = 1 if vol_ratio < 0.8 else (-1 if vol_ratio > 1.2 else 0)
        ret_score = 1 if ret_ratio > 1.2 else (-1 if ret_ratio < 0.8 else 0)
        dd_score = -1 if current_dd < -0.1 else (1 if current_dd > -0.02 else 0)
        
        total_score = vol_score + ret_score + dd_score
        
        # Classify regime
        if total_score >= 2:
            return 'low_vol_up'      # Most favorable
        elif total_score == 1:
            return 'high_vol_up'     # Moderately favorable
        elif total_score == -1:
            return 'low_vol_down'    # Moderately unfavorable
        else:
            return 'high_vol_down'   # Most unfavorable