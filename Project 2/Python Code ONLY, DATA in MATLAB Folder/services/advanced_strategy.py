import numpy as np
from services.advanced_estimators import AdvancedEstimators
from services.advanced_optimization import AdvancedOptimization, risk_parity_weights

class AdvancedStrategy:
    def __init__(self, 
                 lookback_period=36,
                 max_weight=0.05,  # Reduced to 5%
                 target_vol=0.06,  # Target 6% annualized volatility
                 use_pca=True,
                 use_lasso=True,
                 use_shrinkage=True):
        """
        Initialize the advanced strategy with configurable parameters
        
        Parameters:
        -----------
        lookback_period : int
            Number of months to use for estimation
        max_weight : float
            Maximum weight for any single asset
        target_vol : float
            Target portfolio volatility
        use_pca : bool
            Whether to use PCA for factor reduction
        use_lasso : bool
            Whether to use LASSO for factor selection
        use_shrinkage : bool
            Whether to use covariance shrinkage
        """
        self.lookback_period = lookback_period
        self.max_weight = max_weight
        self.target_vol = target_vol
        self.use_pca = use_pca
        self.use_lasso = use_lasso
        self.use_shrinkage = use_shrinkage
        
        # Initialize estimators
        self.estimators = AdvancedEstimators()
        self.optimizer = AdvancedOptimization()
        self.prev_weights = None  # For turnover penalty
        
    def execute_strategy(self, returns, factRet):
        """
        Execute the investment strategy: 80% min variance, 20% risk parity, robust mean, turnover penalty
        """
        # Get covariance matrix using shrinkage
        Q = self.estimators.shrinkage_covariance(returns)
        
        # Use robust mean estimation
        mu = self.estimators.robust_mean(returns, method='winsorized')
        
        # Get risk parity weights
        x_rp = risk_parity_weights(Q, max_weight=self.max_weight)
        
        # Get minimum variance weights
        x_mv = self.optimizer.min_variance(Q, max_weight=self.max_weight)
        
        # Blend: 80% min variance, 20% risk parity
        x = 0.8 * x_mv + 0.2 * x_rp
        
        # Turnover penalty: discourage large changes from previous weights
        if self.prev_weights is not None:
            x = 0.9 * x + 0.1 * self.prev_weights
        
        # Cap maximum position
        x = np.minimum(x, self.max_weight)
        x = x / np.sum(x)  # Renormalize
        
        # Volatility targeting (enforced after all constraints)
        portf_vol = np.sqrt(np.dot(x.T, np.dot(Q, x)) * 12)  # Annualized volatility
        if portf_vol > 0:
            x = x * (self.target_vol / portf_vol)
        x = x / np.sum(np.abs(x))
        
        # Store for next period
        self.prev_weights = x.copy()
        return x 