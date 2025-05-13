import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import linalg

class AdvancedEstimators:
    @staticmethod
    def lasso_factor_model(returns, factRet, alpha=0.1):
        """
        Implements LASSO regression for factor selection and estimation
        """
        T, p = factRet.shape
        X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)
        
        # Fit LASSO for each asset
        lasso = LassoCV(cv=5, random_state=42)
        B = np.zeros((p + 1, returns.shape[1]))
        
        for i in range(returns.shape[1]):
            lasso.fit(X, returns.iloc[:, i])
            B[:, i] = lasso.coef_
        
        # Separate B into alpha and betas
        a = B[0, :]
        V = B[1:, :]
        
        # Calculate residuals and their variance
        ep = returns - X @ B
        sigma_ep = 1 / (T - p - 1) * np.sum(ep.pow(2), axis=0)
        D = np.diag(sigma_ep)
        
        # Factor expected returns and covariance matrix
        f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
        F = factRet.cov().values
        
        # Calculate the asset expected returns and covariance matrix
        mu = np.expand_dims(a, axis=1) + V.T @ f_bar
        Q = V.T @ F @ V + D
        
        # Ensure symmetry
        Q = (Q + Q.T) / 2
        
        return mu, Q, V

    @staticmethod
    def pca_factor_model(returns, factRet, n_components=3):
        """
        Implements PCA-based factor model
        """
        T, p = factRet.shape
        
        # Standardize factor returns
        scaler = StandardScaler()
        factRet_scaled = scaler.fit_transform(factRet)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_factors = pca.fit_transform(factRet_scaled)
        
        # Create design matrix with PCA factors
        X = np.concatenate([np.ones([T, 1]), pca_factors], axis=1)
        
        # OLS regression with PCA factors
        B = np.linalg.solve(X.T @ X, X.T @ returns)
        
        # Separate B into alpha and betas
        a = B[0, :]
        V = B[1:, :]
        
        # Calculate residuals and their variance
        ep = returns - X @ B
        sigma_ep = 1 / (T - n_components - 1) * np.sum(ep.pow(2), axis=0)
        D = np.diag(sigma_ep)
        
        # Factor expected returns and covariance matrix
        f_bar = np.expand_dims(np.mean(pca_factors, axis=0), 1)
        F = np.cov(pca_factors.T)
        
        # Calculate the asset expected returns and covariance matrix
        mu = np.expand_dims(a, axis=1) + V.T @ f_bar
        Q = V.T @ F @ V + D
        
        # Ensure symmetry
        Q = (Q + Q.T) / 2
        
        return mu, Q, V

    @staticmethod
    def shrinkage_covariance(returns):
        """Calculate shrinkage covariance matrix"""
        T = len(returns)
        sample_cov = returns.cov()
        
        # Calculate shrinkage target (constant correlation)
        n = sample_cov.shape[0]
        sample_cov_np = sample_cov.values
        avg_var = float(np.mean(np.diag(sample_cov_np)))
        avg_corr = float((np.sum(sample_cov_np) - np.sum(np.diag(sample_cov_np))) / (n * (n-1)))
        target = avg_var * (avg_corr * np.ones((n,n)) + (1-avg_corr) * np.eye(n))
        
        # Calculate shrinkage intensity
        kappa = float(np.sum((returns.values - returns.values.mean())**4)) / (T * np.sum(sample_cov_np**2))
        delta = max(0, min(1, kappa / T))
        
        # Apply shrinkage
        shrunk_cov = delta * target + (1-delta) * sample_cov
        
        return shrunk_cov

    @staticmethod
    def robust_mean(returns, method='winsorized', alpha=0.1):
        """
        Implements robust mean estimation
        """
        if method == 'winsorized':
            # Winsorized mean
            returns_sorted = np.sort(returns, axis=0)
            k = int(alpha * len(returns))
            returns_sorted[:k] = returns_sorted[k]
            returns_sorted[-k:] = returns_sorted[-k-1]
            mu = np.mean(returns_sorted, axis=0)
        else:
            # Trimmed mean
            mu = np.mean(returns, axis=0)
        
        return np.expand_dims(mu, axis=1) 