function r = adjR2_single(y, yhat, k)
% adjR2_single  Adjusted R² for one regression
%   y      : actual (T×1)
%   yhat   : fitted (T×1)
%   k      : number of estimated parameters (incl. intercept)
%
%   r      : scalar adjusted R²
    T   = length(y);
    SSE = sum( (y - yhat).^2 );
    SST = sum( (y - mean(y)).^2 );
    R2  = 1 - SSE/SST;
    r   = 1 - (1-R2)*(T-1)/(T-k);
end
