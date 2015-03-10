clear;clc;

% path
addpath('auxiliary');
addpath('common');
addpath(genpath('CTBN'));
addpath('liblinear-weights-1.94/matlab');  %% !! Instance-weighted Liblinear 1.94 or above is required !!

% init global var
global LR_implementation;
LR_implementation = 'liblinear';  % designate the library to use (for logistic regression)


% sample run
dataset_name = 'flags';
load(['data/' dataset_name '.mat']);

fprintf('[Training & testing MC on ''%s'']\n', dataset_name);

% 10-fold cv
K = 10;
CVO = cvpartition(Y(:,1), 'kfold', K);

MC = cell(1, K);
Y_pred_MC = cell(1, K);
Y_log_prob_MC = cell(1, K);

for r = 1:CVO.NumTestSets
    fprintf('msg: round %d/%d... ', r, CVO.NumTestSets);
    tic;
    X_tr = X(CVO.training(r), :);
    Y_tr = Y(CVO.training(r), :);
    
    X_ts = X(CVO.test(r), :);
    Y_ts = Y(CVO.test(r), :);
    
    % MC [Hong, Batal, Hauskrecht 2014]
    
    % init MT params
    nCTBN = 10;
    
    % train
    MC_model = train_MC(X_tr, Y_tr, nCTBN);
    
    % test
    [ Y_pred_MC{r}, Y_log_prob_MC{r}] = predict_MC(MC_model, X_ts, Y_ts);
    toc;
    
    % bookkeeping
    MC{r} = getMeasuresMLC(Y_ts, Y_pred_MC{r}, Y_log_prob_MC{r});
end

% report results
fprintf('\n[Test results on ''%s'']\n', dataset_name);
process_results(MC);

