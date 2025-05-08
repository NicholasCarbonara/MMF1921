%% MMF1921 (Summer 2025) - Project 1
%
% Factor models   : OLS, FF3, LASSO, Best‑Subset Selection
% Portfolio stage : estimate μ & Σ → MVO → out‑of‑sample wealth
%
% Student Name :
% Student ID   :

clc
clear all
format short

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Read input files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the stock weekly prices -------------------------------------------
adjClose = readtable('MMF1921_AssetPrices.csv');
adjClose.Properties.RowNames = cellstr(datetime(adjClose.Date));
adjClose.Properties.RowNames = cellstr(datetime(adjClose.Properties.RowNames));
adjClose.Date = [];

% Load the factors weekly returns ----------------------------------------
factorRet = readtable('MMF1921_FactorReturns.csv');
factorRet.Properties.RowNames = cellstr(datetime(factorRet.Date));
factorRet.Properties.RowNames = cellstr(datetime(factorRet.Properties.RowNames));
factorRet.Date = [];

riskFree  = factorRet(:,9);
factorRet = factorRet(:,1:8);

% Identify tickers & dates ------------------------------------------------
tickers = adjClose.Properties.VariableNames';
dates   = datetime(factorRet.Properties.RowNames);

% Compute weekly EXCESS returns ------------------------------------------
prices  = table2array(adjClose);
returns = ( prices(2:end,:) - prices(1:end-1,:) ) ./ prices(1:end-1,:);
returns = returns - ( diag( table2array(riskFree) ) * ones( size(returns) ) );
returns = array2table(returns);
returns.Properties.VariableNames = tickers;
returns.Properties.RowNames = cellstr(datetime(factorRet.Properties.RowNames));

% Align price table to returns/factors (drop first obs.) ------------------
adjClose = adjClose(2:end,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. Initial parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

initialVal = 100000;               % budget ($)
calStart   = datetime('2008-01-01');
calEnd     = calStart + calyears(4) - days(1);

testStart  = datetime('2012-01-01');
testEnd    = testStart + calyears(1) - days(1);

NoPeriods  = 5;                    % yearly rebalances

FMList     = {'OLS','FF','LASSO','BSS'};
FMList     = cellfun(@str2func, FMList, 'UniformOutput', false);
NoModels   = length(FMList);
tags       = {'OLS portfolio','FF portfolio','LASSO portfolio','BSS portfolio'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. Construct & rebalance portfolios
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

toDay      = 0;                    % path index counter (wealth series)
currentVal = zeros(NoPeriods, NoModels);
adjR2      = zeros(NoPeriods, NoModels);
weights    = cell(NoModels,1);

% Hyper‑parameter search grids -------------------------------------------
lambdaGrid = logspace(-4,1,40);    % LASSO λ candidates
kGrid      = 1:3;                  % BSS   K candidates

R2_lasso_grid = NaN(length(lambdaGrid), NoPeriods);
R2_bss_grid   = NaN(length(kGrid),    NoPeriods);

% Containers for chosen λ / K each period --------------------------------
lambda_opt = NaN(NoPeriods,1);
k_opt      = NaN(NoPeriods,1);

fromDay = 1;   % pointer into wealth vector

for t = 1:NoPeriods
    
    % ------- CALIBRATION & TEST windows ---------------------------------
    periodReturns = table2array( returns( calStart <= dates & dates <= calEnd, :) );
    periodFactRet = table2array( factorRet( calStart <= dates & dates <= calEnd, :) );
    currentPrices = table2array( adjClose( ( calEnd - days(7) ) <= dates ...
                                                    & dates <= calEnd, :) )';
    periodPrices  = table2array( adjClose( testStart <= dates & dates <= testEnd,:) );
    
    fromDay = toDay + 1;
    toDay   = toDay + size(periodPrices,1);
    
    % ------- initial wealth ---------------------------------------------
    if t == 1
        currentVal(t,:) = initialVal;
    else
        for i = 1:NoModels
            currentVal(t,i) = currentPrices' * NoShares{i};
        end
    end
    
    % ------- parameter estimation ---------------------------------------
    for i = 1:NoModels
        
        modelName = func2str(FMList{i});
        
        switch modelName
            
            % ============================================================= LASSO
            case 'LASSO'
                bestR2 = -Inf;
                m      = size(periodReturns,1);
                foldSz = floor(m/4);
                folds  = arrayfun(@(k) ((k-1)*foldSz+1):(k*foldSz), 1:4, 'Uni',0);
                
                for l = 1:length(lambdaGrid)
                    lam = lambdaGrid(l);
                    R2_folds = zeros(1,4);
                    
                    for f = 1:4
                        testIdx  = folds{f};
                        trainIdx = setdiff(1:m, testIdx);
                        
                        Xtr = [ones(length(trainIdx),1), periodFactRet(trainIdx,:)];
                        Xte = [ones(length(testIdx),1),  periodFactRet(testIdx,:) ];
                        
                        R2_assets = zeros(1, size(periodReturns,2));
                        
                        for j = 1:size(periodReturns,2)
                            ytr = periodReturns(trainIdx,j);
                            yte = periodReturns(testIdx ,j);
                            
                            d   = size(Xtr,2);
                            A   = [eye(d), -eye(d)];          % β⁺/β⁻ split
                            H   = (2/length(ytr)) * ( (Xtr*A)' * (Xtr*A) );
                            f_q = lam*ones(2*d,1) - (2/length(ytr))*( (Xtr*A)'*ytr );
                            
                            z   = quadprog(H, f_q, [], [], [], [], ...
                                           zeros(2*d,1), [], [], ...
                                           optimoptions('quadprog','Display','off'));
                            beta = z(1:d) - z(d+1:end);
                            
                            yhat = Xte * beta;
                            R2_assets(j) = 1 - sum((yte - yhat).^2) ...
                                             / sum((yte - mean(yte)).^2);
                        end
                        R2_folds(f) = mean(R2_assets);
                    end
                    
                    R2_lasso_grid(l,t) = mean(R2_folds);
                    
                    if R2_lasso_grid(l,t) > bestR2
                        bestR2         = R2_lasso_grid(l,t);
                        [mu{i}, Q{i}]  = LASSO(periodReturns, periodFactRet, lam, 0);
                        lambda_opt(t)  = lam;
                    end
                end
                
            % ============================================================= BSS
            case 'BSS'
                bestR2 = -Inf;
                
                for kval = kGrid
                    [muTmp,QTmp] = BSS(periodReturns, periodFactRet, 0, kval);
                    
                    X      = [ones(size(periodFactRet,1),1), periodFactRet(:,1:kval)];
                    RsMean = getMeanAdjR2(X, periodReturns);
                    
                    if RsMean > bestR2
                        bestR2      = RsMean;
                        mu{i}       = muTmp;
                        Q{i}        = QTmp;
                        k_opt(t)    = kval;
                    end
                    R2_bss_grid(kval,t) = RsMean;
                end
            
            % ============================================================= OLS / FF
            otherwise
                [mu{i}, Q{i}] = FMList{i}(periodReturns, periodFactRet, 0, 0);
        end
        
        % ------- adjusted R² (store mean across assets) ------------------
        switch modelName
            case 'OLS'
                X = [ones(size(periodFactRet,1),1), periodFactRet];
                k_pred = 9;
            case 'FF'
                X = [ones(size(periodFactRet,1),1), periodFactRet(:,1:3)];
                k_pred = 4;
            case 'LASSO'
                X = [ones(size(periodFactRet,1),1), periodFactRet];
                k_pred = 9;
            case 'BSS'
                X = [ones(size(periodFactRet,1),1), periodFactRet(:,1:k_opt(t))];
                k_pred = k_opt(t) + 1;
        end
        
        Rs = zeros(size(periodReturns,2),1);
        for a = 1:size(periodReturns,2)
            y     = periodReturns(:,a);
            beta  = X\y;
            Rs(a) = calc_adjR2(y, X*beta, k_pred);
        end
        adjR2(t,i) = mean(Rs);
    end
    
    % ------- report best λ / K for this period --------------------------
    if ~isnan(lambda_opt(t))
        fprintf('✓ Period %d  |  optimal λ  = %.4g\n', t, lambda_opt(t));
    end
    if ~isnan(k_opt(t))
        fprintf('✓ Period %d  |  optimal K  = %d\n',   t, k_opt(t));
    end
    
    % ------- MVO optimisation -------------------------------------------
    for i = 1:NoModels
        targetRet = geomean(periodFactRet(:,1)+1)-1;
        x{i}(:,t) = MVO(mu{i}, Q{i}, targetRet);
        weights{i}= x{i};
    end
    
    % ------- shares held & wealth path ----------------------------------
    for i = 1:NoModels
        NoShares{i} = x{i}(:,t) .* currentVal(t,i) ./ currentPrices;
        
        assetRets = (periodPrices(2:end,:) - periodPrices(1:end-1,:)) ...
                                          ./ periodPrices(1:end-1,:);
        assetRets = [zeros(1,size(assetRets,2)); assetRets];
        portfRets = assetRets * NoShares{i} ./ sum(NoShares{i});
        
        portfValue(fromDay,i) = currentVal(t,i);
        for k = 2:length(portfRets)
            portfValue(fromDay+k-1,i) = portfValue(fromDay+k-2,i) ...
                                      * (1 + portfRets(k));
        end
    end
    
    % ------- roll windows ------------------------------------------------
    calStart  = calStart  + calyears(1);
    calEnd    = calStart  + calyears(4) - days(1);
    testStart = testStart + calyears(1);
    testEnd   = testStart + calyears(1) - days(1);
end  % end main loop

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ------------------------------------------------------------------ 4.0 Hyper‑parameter summary
fprintf('\n================  Hyper‑parameter Summary  ================\n');
fprintf('Period   Optimal λ (LASSO)   Optimal K (BSS)\n');
fprintf('------   -----------------   ---------------\n');
for t = 1:NoPeriods
    fprintf('%6d   %15.6g   %15s\n', ...
        t, ...
        lambda_opt(t), ...
        ternary(~isnan(k_opt(t)), num2str(k_opt(t)), '‑'));
end
fprintf('============================================================\n\n');

% ------------------------------------------------------------------ 4.1 Adjusted R² table
rowNames   = strcat("Period_", string(1:NoPeriods));
adjR2Table = array2table(adjR2, 'RowNames', rowNames, ...
                                   'VariableNames', tags);
disp('Adjusted R^2 (In‑Sample) per Model and Period:');
disp(adjR2Table);

% ------------------------------------------------------------------ 4.2 Performance metrics
portfRet = diff(portfValue)./portfValue(1:end-1,:);
avgRet   = mean(portfRet);
vol      = std (portfRet);
sharpe   = avgRet ./ vol;

annFactor = sqrt(12);
annRet    = (1+avgRet).^12 - 1;
annVol    = vol * annFactor;
annSharpe = sharpe * annFactor;

fprintf('\n*** Out‑of‑sample performance (annualised) 2012‑2016 ***\n');
fprintf('%-10s  %8s  %8s  %8s\n','Model','Return','St.dev','Sharpe');
for i = 1:NoModels
    fprintf('%-10s  %8.2f  %8.2f  %8.2f\n', ...
        tags{i}, 100*annRet(i), 100*annVol(i), annSharpe(i));
end

% ------------------------------------------------------------------ 4.3 Wealth plot
plotDates = dates(dates >= datetime('2012-01-01'));
figure(1); clf
for i = 1:NoModels
    plot(plotDates, portfValue(:,i)); hold on
end
legend(tags,'Location','eastoutside','FontSize',12);
datetick('x','dd-mmm-yyyy','keepticks','keeplimits');
set(gca,'XTickLabelRotation',30);
title('Portfolio value','FontSize',14);
ylabel('Value','FontSize',12);
print(gcf,'fileName','-dpng','-r0');

% ------------------------------------------------------------------ 4.4 Portfolio‑weights area charts
for i = 1:NoModels
    figure(i+1); clf
    area(x{i}'); box on
    legend(tickers,'Location','eastoutside','FontSize',10);
    title([tags{i} ' weights'],'FontSize',14);
    ylabel('Weights'); xlabel('Rebalance period');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions (all referenced above)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function R2_mean = getMeanAdjR2(X, Y)
    [T,n] = size(Y); k = size(X,2);
    Rs = zeros(n,1);
    for i = 1:n
        y    = Y(:,i);
        beta = X\y;
        yhat = X*beta;
        Rs(i)= calc_adjR2(y,yhat,k);
    end
    R2_mean = mean(Rs);
end

function r = calc_adjR2(y,yhat,k)
    T   = length(y);
    SSE = sum((y - yhat).^2);
    SST = sum((y - mean(y)).^2);
    R2  = 1 - SSE/SST;
    r   = 1 - (1 - R2)*(T - 1)/(T - k);
end

function out = ternary(cond,a,b)
    if cond; out = a; else; out = b; end
end