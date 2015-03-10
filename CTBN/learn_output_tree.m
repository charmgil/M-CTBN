function [ T ] = learn_output_tree( X, Y, is_switching )

is_profiling = true;

% param setting
if ~exist('is_switching', 'var');
    is_switching = false;
end


% init
[n, m]=size(Y);


% proc
if is_profiling, t1 = clock; end;

%3-folds internal cross validation
k=3;
%rand('seed',1);
indices = crossvalind('Kfold',Y(:,1),k);
%load('idx_cv.mat');

idx=1;
S=[];
for i=1:m
    if mod(i, 10) == 0
        fprintf('...%d', i);
    end
    [ LL_without_Y ] = compute_crossvalidation_loglikelihood( X, Y(:,i), indices, k);
    node_weights(i) = LL_without_Y;
    for j = 1:m 
        
        if(j ~= i)
            % measure the influence of Y_j on Y_i
            if ~is_switching
                [ LL_with_y ] = compute_crossvalidation_loglikelihood( [X Y(:,j)], Y(:,i), indices, k);
            else
                [ LL_with_y ] = compute_crossvalidation_loglikelihood_sw( X, Y(:,i), Y(:,j), indices, k);
            end

            %condition
            if(LL_with_y > LL_without_Y)
                S{idx}.from = j;
                S{idx}.to = i;
                %S{idx}.weight = LL_with_y-LL_without_Y;
                S{idx}.weight = LL_with_y;
                idx = idx+1;
            end
        end
        
    end
end

if is_profiling, t2 = clock; end;

for i=1:m
    [T_list{i}, T_weight(i)] = directed_maximum_spanning_tree(S, 1:m, i, node_weights);
end

[max_weight, max_idx] = max(T_weight);

T = T_list{max_idx};

if is_profiling, t3 = clock; end;

[ T ] = organize_tree_BFS(T);

if is_profiling, t4 = clock; end;

if ~is_switching
    T = compute_tree_weights(T, X, Y);
else
    T = compute_tree_weights(T, X, Y, '-s 1 -c 1');
end

if is_profiling, t5 = clock; end;

if is_profiling,
    fprintf( '(pf)for-compute_edge_weight: %f s\n', etime(t2,t1) );
    fprintf( '(pf)directed_maximum_spanning_tree: %f s\n', etime(t3,t2) );
    fprintf( '(pf)organize_tree_BFS: %f s\n', etime(t4,t3) );
    fprintf( '(pf)compute_tree_weights: %f s\n', etime(t5,t4) );
end;
