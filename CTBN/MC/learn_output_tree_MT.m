% Mixture of CTBN, EM algorithm
function [Trees, lambda, gamma, lambda_over_time, Q_over_time, LL_train_over_time, LL_test_over_time, t_testing] = learn_output_tree_MT(X, Y, Trees, is_switching, X_test, Y_test, max_iter, init_gamma )

global LR_implementation;

is_init_train = false;  % init_train: true, if initial model train is required (won't be necessary if boosting is ready)

if ~exist('is_switching','var')
    is_switching = false;
end

if exist('X_test','var') && ~isempty(X_test) % test on fly? :: Debug purpose only. Test set should be supplied as X_test, Y_test
    is_testing = true;
else
    is_testing = false;
end


lambda_over_time = [];
Q_over_time = [];
LL_train_over_time = [];
LL_test_over_time = [];

t_testing = 0;

K = length(Trees);
[N, d] = size(Y);

TOL = 1e-3;

if ~exist('max_iter','var')
    MAXITER = 250;
else
    MAXITER = max_iter;
end


% parallelize (divide batch sets)
global runParallel;
global num_batches;

if runParallel
    X_temp = cell(1,num_batches);
    Y_temp = cell(1,num_batches);
    
    batch_size = round(N/num_batches);
    batch_size = repmat(batch_size, [1 num_batches]);
    
    s = 1;
    e = s+batch_size(1)-1;
    for i = 1:num_batches
        par_idx{i} = (s:e)';
        X_temp{i} = X(s:e,:);
        Y_temp{i} = Y(s:e,:);

        % prep next
        s = e+1;
        e = s+batch_size(i)-1;
        if i == num_batches-1 && e ~= N
            e = N;
            batch_size(num_batches) = e-s+1;
        end
    end
end



% init
if exist('init_gamma','var')
    gamma = init_gamma;
else
    gamma = ones(N,K)/K;
    %gamma = [ones(N,1)*0.3 ones(N,1)*0.7 ];
end
Gamma = sum(gamma);
lambda = sum(gamma)/N;
for k = 1:K
    w(:,k) = gamma(:,k) / Gamma(k);
end
n_iter = 0;

Q = 0;
for k = 1:K
    if is_init_train
        if is_switching
            Trees{k} = compute_tree_weights(Trees{k}, X, Y, '-s 1 -c 1');
        else
            Trees{k} = compute_tree_weights(Trees{k}, X, Y, '-s 0 -c 1');
        end
    end
    
    if runParallel
        ll_temp = zeros(1, num_batches);
        parfor j = 1:num_batches
            ll_temp(j) = ll_temp(j) + compute_loglikelihood(Trees{k}, X_temp{j}, Y_temp{j}, is_switching);
        end
        ll = sum(ll_temp);
    else
        ll = compute_loglikelihood(Trees{k}, X, Y, is_switching);
    end
    
    Q = Q + Gamma(k) * log(lambda(k)) + Gamma(k) * 1/N * ll;
end

% Trees{1}{:}
% Trees{2}{:}



% bookkeeping
lambda_over_time(:,1) = lambda;
Q_over_time(1) = Q;
if is_testing
    t_test_start = clock;
    LL_train_over_time(1) = compute_loglikelihood_MT(Trees, lambda, X, Y, is_switching);
    LL_test_over_time(1) = compute_loglikelihood_MT(Trees, lambda, X_test, Y_test, is_switching);
    %fprintf( '%f / %f / %f\n', Q, LL_train_over_time(1), LL_test_over_time(1) );
    t_test_end = clock;
    t_testing = t_testing + etime(t_test_end,t_test_start);
end
is_converged = false;


while ~is_converged && n_iter < MAXITER
    
    n_iter = n_iter + 1;
    
    % expectation
    if runParallel
        prob = cell(1,num_batches);
        gamma_temp = cell(1,num_batches);
        parfor j = 1:num_batches
            for i = 1:batch_size(j)
                for k = 1:K
                    [~,prob{j}] = compute_loglikelihood( Trees{k}, X_temp{j}(i,:), Y_temp{j}(i,:), is_switching );
                    gamma_temp{j}(i,k) = prob{j} * lambda(k);
                end
                %normalize
                gamma_temp{j}(i,:) = bsxfun(@rdivide,gamma_temp{j}(i,:),sum(gamma_temp{j}(i,:),2));
            end
        end
        
        gamma = [];
        for j = 1:num_batches
            gamma = [ gamma; gamma_temp{j} ];
        end
        
    else
        prob = [];
        for i = 1:N
            for k = 1:K
                [~,prob] = compute_loglikelihood( Trees{k}, X(i,:), Y(i,:), is_switching );
                gamma(i,k) = prob * lambda(k);
            end
            %normalize
            gamma(i,:) = bsxfun(@rdivide,gamma(i,:),sum(gamma(i,:),2));
        end
    end
    
    
    

    Gamma = sum(gamma,1);
    w = bsxfun(@rdivide,gamma,Gamma);
    lambda = Gamma/N;
    
    % maximization
    % weighted LR train
    for k = 1:K
        if is_switching
            Trees{k} = compute_tree_weights(Trees{k}, X, Y, w(:,k), '-s 1 -l 0');
        else
            Trees{k} = compute_tree_weights(Trees{k}, X, Y, w(:,k), '-s 0 -l 0');
        end
    end
    
    % score Q
    if runParallel
        Q_new_temp = zeros(1,num_batches);
        parfor j = 1:num_batches
            W_temp{j} = w(par_idx{j},:);
            
            for k = 1:K
                sum_wll = 0;
                for i = 1:batch_size(j)
                    ll = compute_loglikelihood( Trees{k}, X_temp{j}(i,:), Y_temp{j}(i,:), is_switching );
                    sum_wll = sum_wll + W_temp{j}(i,k) * ll;
                end
                if j == 1
                    Q_new_temp(j) = Q_new_temp(j) + Gamma(k) * log(lambda(k)) + Gamma(k) * sum_wll;
                else
                    Q_new_temp(j) = Q_new_temp(j) + Gamma(k) * sum_wll;
                end
            end
        end
        Q_new = sum(Q_new_temp);
        
    else
        Q_new = 0;
        
        for k = 1:K
            sum_wll = 0;
            for i = 1:N
                ll = compute_loglikelihood( Trees{k}, X(i,:), Y(i,:), is_switching );
                sum_wll = sum_wll + w(i,k) * ll;
            end
            Q_new = Q_new + Gamma(k) * log(lambda(k)) + Gamma(k) * sum_wll;
        end
    end
    
    if is_testing
        t_test_start = clock;
        LL_train_over_time(n_iter+1) = compute_loglikelihood_MT(Trees, lambda, X, Y, is_switching);
        LL_test_over_time(n_iter+1) = compute_loglikelihood_MT(Trees, lambda, X_test, Y_test, is_switching);
        %fprintf( '%f / %f / %f\n', Q_new, LL_train_over_time(n_iter+1), LL_test_over_time(n_iter+1) );
        t_test_end = clock;
        t_testing = t_testing + etime(t_test_end,t_test_start);
    end

    % continue?
    is_converged = Q_new - Q < TOL;
    if n_iter == 1
        fprintf('(n_CTBNs = %d)\n', K);
        fprintf('init: Q = %.2f\t', Q);
        if is_testing
            fprintf(' | LL_tr: %.2f\t| LL_ts: %.2f\t| lambda: ', LL_train_over_time(n_iter), LL_test_over_time(n_iter));
        end
        fprintf('Lambda = [ ');
        fprintf('%.3f  ', lambda_over_time(:,1));
        fprintf('\b]\n');
    end
    fprintf( '%d(%d): Q = %.2f\t', n_iter, is_converged, Q_new );
    if is_testing
        fprintf(' | LL_tr: %.2f\t| LL_ts: %.2f\t| lambda: ', LL_train_over_time(n_iter+1), LL_test_over_time(n_iter+1) );
    end
    fprintf('Lambda = [ ');
    fprintf('%.3f  ', lambda);
    fprintf('\b]\n');
    
%     gamma(1:10,:)
%     w(1:10,:)
    
%     if n_iter < 3
%         Trees{1}{:}
%         Trees{2}{:}
%     end
    
    Q = Q_new;
    
    % bookkeeping
    lambda_over_time(:,n_iter+1) = lambda;
    Q_over_time(n_iter+1) = Q;
    
end

if is_converged
    fprintf('Converged in %d steps.\n', n_iter);
else
    fprintf('Not converged in %d steps.\n', MAXITER);
end

if is_testing
    fprintf(2,'msg: spent %.2f sec for testing.\n', t_testing );
end

end
