import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import linalg

class AdvancedEstimators:
    def __init__(self, 
                 lookback_period=36,
                 shrinkage_intensity=0.5,
                 use_pca=True,
                 use_lasso=True,
                 use_shrinkage=True):
        self.lookback_period = lookback_period
        self.shrinkage_intensity = shrinkage_intensity
        self.use_pca = use_pca
        self.use_lasso = use_lasso
        self.use_shrinkage = use_shrinkage
        
    def estimate_returns(self, returns, factors=None):
        """
        Enhanced return estimation with multiple models
        """
        if factors is not None and (self.use_pca or self.use_lasso):
            # Factor-based estimation
            if self.use_lasso:
                beta = self._estimate_factor_loadings_lasso(returns, factors)
            else:
                beta = self._estimate_factor_loadings_pca(returns, factors)
                
            # Calculate factor expected returns (historical mean)
            factor_returns = np.mean(factors, axis=0)
            
            # Combine for expected returns
            expected_returns = beta @ factor_returns
        else:
            # Simple historical mean with shrinkage
            historical_mean = np.mean(returns, axis=0)
            if self.use_shrinkage:
                grand_mean = np.mean(historical_mean)
                expected_returns = (1 - self.shrinkage_intensity) * historical_mean + \
                                 self.shrinkage_intensity * grand_mean
            else:
                expected_returns = historical_mean
                
        return expected_returns
        
    def estimate_covariance(self, returns, factors=None):
        """
        Enhanced covariance estimation with shrinkage and factor structure
        """
        sample_cov = np.cov(returns, rowvar=False)
        
        if factors is not None and (self.use_pca or self.use_lasso):
            # Factor-based covariance
            if self.use_lasso:
                beta = self._estimate_factor_loadings_lasso(returns, factors)
            else:
                beta = self._estimate_factor_loadings_pca(returns, factors)
                
            factor_cov = np.cov(factors, rowvar=False)
            systematic_cov = beta @ factor_cov @ beta.T
            
            # Residual variance
            residual_var = np.diag(sample_cov) - np.diag(systematic_cov)
            residual_var = np.maximum(residual_var, 1e-6)  # Ensure positive
            
            # Combine systematic and idiosyncratic components
            covariance = systematic_cov + np.diag(residual_var)
        else:
            covariance = sample_cov
            
        if self.use_shrinkage:
            # Apply shrinkage to identity matrix
            target = np.diag(np.diag(covariance))
            covariance = (1 - self.shrinkage_intensity) * covariance + \
                        self.shrinkage_intensity * target
                        
        # Ensure positive definiteness
        min_eigenval = np.min(np.linalg.eigvals(covariance))
        if min_eigenval < 1e-6:
            covariance += (abs(min_eigenval) + 1e-6) * np.eye(covariance.shape[0])
            
        return covariance
        
    def _estimate_factor_loadings_lasso(self, returns, factors, alpha=0.01):
        """
        Estimate factor loadings using LASSO regression
        """
        from sklearn.linear_model import LassoCV
        
        n_assets = returns.shape[1]
        beta = np.zeros((n_assets, factors.shape[1]))
        
        for i in range(n_assets):
            model = LassoCV(cv=5, random_state=42)
            model.fit(factors, returns[:, i])
            beta[i, :] = model.coef_
            
        return beta
        
    def _estimate_factor_loadings_pca(self, returns, factors, n_components=None):
        """
        Estimate factor loadings using PCA
        """
        from sklearn.decomposition import PCA
        
        if n_components is None:
            n_components = min(factors.shape[1], returns.shape[1])
            
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(returns)
        
        return pca.components_.T

    @staticmethod
    def lasso_factor_model(returns, factRet, alpha=0.1):
        """
        Implements LASSO regression for factor selection and estimation with improved accuracy
        """
        T, p = factRet.shape
        
        # Convert pandas objects to numpy arrays
        returns_np = returns.values if hasattr(returns, 'values') else returns
        factRet_np = factRet.values if hasattr(factRet, 'values') else factRet
        
        # Standardize inputs for better numerical stability
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X = np.concatenate([np.ones([T, 1]), factRet_np], axis=1)
        X_scaled = np.concatenate([np.ones([T, 1]), scaler_X.fit_transform(factRet_np)], axis=1)
        
        # Adjust CV folds based on sample size
        n_splits = min(5, max(2, T // 3))
        
        # Enhanced LASSO with multiple alphas and time-weighted observations
        alphas = np.logspace(-4, 0, 50)  # More granular alpha grid
        
        # Time weights for more recent observations
        time_weights = 0.97 ** np.arange(T-1, -1, -1)
        time_weights = time_weights / np.sum(time_weights)
        
        # Fit LASSO for each asset with improved settings
        lasso = LassoCV(
            cv=n_splits,
            random_state=42,
            max_iter=5000,
            tol=1e-4,
            selection='random',
            n_jobs=-1,
            alphas=alphas
        )
        
        # Ridge as fallback with multiple alphas
        ridge = RidgeCV(
            cv=n_splits,
            fit_intercept=False,
            alphas=np.logspace(-4, 4, 50)
        )
        
        B = np.zeros((p + 1, returns.shape[1]))
        
        for i in range(returns.shape[1]):
            y = returns_np[:, i]
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
            
            # Try LASSO first
            try:
                # Apply time weights
                sample_weights = time_weights * np.sqrt(T)  # Scale weights for numerical stability
                lasso.fit(X_scaled[:, 1:], y_scaled, sample_weight=sample_weights)
                coef = lasso.coef_
                intercept = scaler_y.mean_ - np.sum(coef * scaler_X.mean_ / scaler_X.scale_)
                B[0, i] = intercept
                B[1:, i] = coef / scaler_X.scale_
            except:
                # Fallback to Ridge if LASSO fails
                ridge.fit(X, y)
                B[:, i] = ridge.coef_
        
        # Calculate residuals and their variance with improved regularization
        ep = returns_np - X @ B
        
        # Time-weighted residual variance
        sigma_ep = np.sum(time_weights.reshape(-1, 1) * ep ** 2, axis=0)
        D = np.diag(sigma_ep)
        
        # Enhanced expected returns estimation
        f_bar = np.sum(time_weights.reshape(-1, 1) * factRet_np, axis=0).reshape(-1, 1)
        raw_F = np.cov(factRet_np.T, aweights=time_weights)
        
        # Adaptive shrinkage based on sample size and condition number
        shrinkage = min(0.5, 1 - T/(2*p))
        cond_num = np.linalg.cond(raw_F)
        if cond_num > 100:
            shrinkage = max(shrinkage, 0.5)  # More aggressive shrinkage for ill-conditioned matrices
        
        F = (1 - shrinkage) * raw_F + shrinkage * np.diag(np.diag(raw_F))
        
        # Calculate expected returns with robust estimation
        mu_hist = np.sum(time_weights.reshape(-1, 1) * returns_np, axis=0).reshape(-1, 1)
        mu_fact = B[1:, :].T @ f_bar + B[0, :].reshape(-1, 1)
        
        # Adaptive combination based on R-squared
        r_squared = 1 - np.sum(ep ** 2, axis=0) / np.sum((returns_np - np.mean(returns_np, axis=0)) ** 2, axis=0)
        alpha_combine = np.clip(r_squared, 0.25, 0.75)  # Limit the range of combination
        
        mu = alpha_combine.reshape(-1, 1) * mu_fact + (1 - alpha_combine.reshape(-1, 1)) * mu_hist
        
        # Covariance matrix with factor structure and enhanced stability
        Q = B[1:, :].T @ F @ B[1:, :] + D
        
        # Ensure symmetry and positive definiteness
        Q = (Q + Q.T) / 2
        min_eig = np.min(np.real(linalg.eigvals(Q)))
        if min_eig < 1e-8:
            Q += (abs(min_eig) + 1e-8) * np.eye(Q.shape[0])
        
        return mu, Q, B[1:, :].T

    @staticmethod
    def pca_factor_model(returns, factRet, n_components=None):
        """
        Implements PCA-based factor model with improved stability
        """
        T, p = factRet.shape
        
        # Adaptive number of components based on sample size
        if n_components is None:
            n_components = min(3, p // 2)  # Use at most half the number of factors
        
        # Standardize factor returns
        scaler = StandardScaler()
        factRet_scaled = scaler.fit_transform(factRet)
        
        # Apply PCA with whitening
        pca = PCA(n_components=n_components, whiten=True)
        pca_factors = pca.fit_transform(factRet_scaled)
        
        # Create design matrix with PCA factors
        X = np.concatenate([np.ones([T, 1]), pca_factors], axis=1)
        
        # OLS regression with regularization
        XtX = X.T @ X
        XtX_reg = XtX + 1e-8 * np.eye(XtX.shape[0])  # Add small regularization
        B = np.linalg.solve(XtX_reg, X.T @ returns)
        
        # Separate B into alpha and betas
        a = B[0, :]
        V = B[1:, :]
        
        # Calculate residuals and their variance with regularization
        ep = returns - X @ B
        sigma_ep = 1 / (T - n_components - 1 + 1e-8) * np.sum(ep.pow(2), axis=0)
        D = np.diag(sigma_ep)
        
        # Factor expected returns and covariance matrix with shrinkage
        f_bar = np.expand_dims(np.mean(pca_factors, axis=0), 1)
        raw_F = np.cov(pca_factors.T)
        
        # Adaptive shrinkage based on sample size
        shrinkage = min(0.5, 1 - T/(2*n_components))
        F = (1 - shrinkage) * raw_F + shrinkage * np.diag(np.diag(raw_F))
        
        # Calculate the asset expected returns and covariance matrix
        mu = np.expand_dims(a, axis=1) + V.T @ f_bar
        Q = V.T @ F @ V + D
        
        # Ensure symmetry and positive definiteness
        Q = (Q + Q.T) / 2
        min_eig = np.min(np.real(linalg.eigvals(Q)))
        if min_eig < 1e-8:
            Q += (abs(min_eig) + 1e-8) * np.eye(Q.shape[0])
        
        return mu, Q, V

    @staticmethod
    def shrinkage_covariance(returns):
        """
        Calculate shrinkage covariance matrix with improved stability
        """
        T = len(returns)
        sample_cov = returns.cov()
        
        # Calculate shrinkage target (constant correlation)
        n = sample_cov.shape[0]
        sample_cov_np = sample_cov.values
        avg_var = float(np.mean(np.diag(sample_cov_np)))
        
        # More robust correlation estimation
        corr_mat = sample_cov_np / np.sqrt(np.outer(np.diag(sample_cov_np), np.diag(sample_cov_np)))
        avg_corr = float((np.sum(corr_mat) - n) / (n * (n-1)))  # Exclude diagonal
        
        # Ensure correlation is in [-1, 1]
        avg_corr = max(-1, min(1, avg_corr))
        
        target = avg_var * (avg_corr * np.ones((n,n)) + (1-avg_corr) * np.eye(n))
        
        # Adaptive shrinkage intensity based on sample size
        shrinkage = min(0.9, 1 - T/(2*n))  # More aggressive shrinkage for small samples
        
        # Apply shrinkage
        shrunk_cov = shrinkage * target + (1-shrinkage) * sample_cov
        
        # Ensure positive definiteness
        min_eig = np.min(np.real(linalg.eigvals(shrunk_cov)))
        if min_eig < 1e-8:
            shrunk_cov += (abs(min_eig) + 1e-8) * np.eye(n)
        
        return shrunk_cov

    @staticmethod
    def robust_mean(returns, method='winsorized', alpha=0.1):
        """
        Implements robust mean estimation with improved handling of outliers
        """
        if method == 'winsorized':
            # Adaptive winsorization based on sample size
            T = len(returns)
            alpha = min(alpha, 0.25 * (1 - np.exp(-T/60)))  # Reduce winsorization for small samples
            
            # Winsorized mean
            returns_sorted = np.sort(returns, axis=0)
            k = max(1, int(alpha * len(returns)))  # At least 1 point
            returns_sorted[:k] = returns_sorted[k]
            returns_sorted[-k:] = returns_sorted[-k-1]
            mu = np.mean(returns_sorted, axis=0)
        else:
            # Trimmed mean with minimum sample size check
            if len(returns) < 10:
                mu = np.mean(returns, axis=0)  # Use regular mean for very small samples
            else:
                # Adaptive trimming
                trim_ratio = min(0.1, 0.4 * (1 - np.exp(-len(returns)/60)))
                k = int(trim_ratio * len(returns))
                mu = np.mean(returns.sort_values().iloc[k:-k], axis=0)
        
        return np.expand_dims(mu, axis=1) 

    @staticmethod
    def robust_return_estimation(returns, factRet, alpha=0.1):
        """
        Ensemble approach for robust return estimation combining multiple methods
        """
        # Convert inputs to numpy arrays if needed
        returns_np = returns.values if hasattr(returns, 'values') else returns
        factRet_np = factRet.values if hasattr(factRet, 'values') else factRet
        
        # Calculate market conditions
        recent_vol = np.std(returns_np[-12:]) * np.sqrt(252)
        long_vol = np.std(returns_np) * np.sqrt(252)
        vol_ratio = recent_vol / long_vol
        
        recent_ret = np.mean(returns_np[-12:]) * 252
        long_ret = np.mean(returns_np) * 252
        ret_ratio = recent_ret / (long_ret + 1e-8)
        
        # Determine market regime
        favorable_conditions = (vol_ratio < 1.2 and ret_ratio > 1.0) or (recent_ret > 0 and vol_ratio < 0.8)
        
        # 1. LASSO-based estimation with adaptive shrinkage
        mu_lasso, Q_lasso, _ = AdvancedEstimators.lasso_factor_model(returns, factRet)
        
        # 2. PCA-based estimation
        mu_pca, Q_pca, _ = AdvancedEstimators.pca_factor_model(returns, factRet)
        
        # 3. Robust mean estimation with reduced shrinkage in favorable conditions
        if favorable_conditions:
            alpha = alpha * 0.5  # Less shrinkage in favorable conditions
        mu_robust = AdvancedEstimators.robust_mean(returns, method='winsorized', alpha=alpha)
        
        # Ensure all estimates are 2D arrays (n x 1)
        mu_lasso = np.array(mu_lasso).reshape(-1, 1)
        mu_pca = np.array(mu_pca).reshape(-1, 1)
        mu_robust = np.array(mu_robust).reshape(-1, 1)
        
        # Calculate information ratios
        ir_lasso = float(np.mean(mu_lasso) / np.sqrt(np.diag(Q_lasso)).mean())
        ir_pca = float(np.mean(mu_pca) / np.sqrt(np.diag(Q_pca)).mean())
        ir_robust = float(np.mean(mu_robust) / np.std(returns_np))
        
        # Adaptive weights based on market conditions
        if favorable_conditions:
            # More weight on aggressive estimates
            w_lasso = 0.5
            w_pca = 0.3
            w_robust = 0.2
        else:
            # More weight on conservative estimates
            total_ir = abs(ir_lasso) + abs(ir_pca) + abs(ir_robust)
            w_lasso = abs(ir_lasso) / total_ir
            w_pca = abs(ir_pca) / total_ir
            w_robust = abs(ir_robust) / total_ir
        
        # Combine estimates
        mu_combined = w_lasso * mu_lasso + w_pca * mu_pca + w_robust * mu_robust
        
        # Adaptive shrinkage based on market conditions
        if favorable_conditions:
            # Less shrinkage in favorable conditions
            mu_mean = np.mean(mu_combined)
            mu_std = np.std(mu_combined)
            mu_combined = np.clip(mu_combined, 
                                mu_mean - 3*mu_std,  # Allow more extreme positive returns
                                mu_mean + 3*mu_std)
        else:
            # More conservative shrinkage
            mu_mean = np.mean(mu_combined)
            mu_std = np.std(mu_combined)
            mu_combined = np.clip(mu_combined, 
                                mu_mean - 2*mu_std,
                                mu_mean + 2*mu_std)
        
        # Adaptive covariance estimation
        if favorable_conditions:
            # More weight on recent data in favorable conditions
            w_cov_stable = 0.3
        else:
            # More weight on stable estimates in unfavorable conditions
            w_cov_stable = 0.7
        
        Q_combined = w_cov_stable * Q_pca + (1 - w_cov_stable) * Q_lasso
        
        # Ensure positive definiteness with adaptive regularization
        min_eig = np.min(np.real(linalg.eigvals(Q_combined)))
        if min_eig < 1e-8:
            reg_factor = 1e-8 if favorable_conditions else 1e-6
            Q_combined += (abs(min_eig) + reg_factor) * np.eye(Q_combined.shape[0])
            
        return mu_combined, Q_combined 