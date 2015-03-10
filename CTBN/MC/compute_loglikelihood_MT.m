%LL: loglikelihood of [X,Y]
%avg_prob: the average probability for the true labels P(yi|xi)
function [ LL, avg_prob, Y_prob ] = compute_loglikelihood_MT( Trees, lambda, X, Y, is_switching )

global runParallel;
global num_batches;

if (~isempty(runParallel) && ~isempty(num_batches)) && runParallel && num_batches <= size(X,1)
    [ LL, avg_prob, Y_prob ] = compute_loglikelihood_MT_parallel( Trees, lambda, X, Y, is_switching );
else
    [ LL, avg_prob, Y_prob ] = compute_loglikelihood_MT_sequential( Trees, lambda, X, Y, is_switching );
end

end%end-of-function compute_loglikelihood_MT()


%%%%
function [ LL, avg_prob, Y_prob ] = compute_loglikelihood_MT_sequential( Trees, lambda, X, Y, is_switching )

K = length(Trees);
n = size(X,1);
Y_prob = zeros(n,1);

for i = 1:n
    for k = 1:K
        [~, tmp_prob(k)] = compute_loglikelihood( Trees{k}, X(i,:), Y(i,:), is_switching );
        %tmp_prob(k) = exp(evaluate_probability( Trees{k}, Y(i,:)) );
    end
    
    Y_prob(i,1) = dot(tmp_prob, lambda);
end

LL = sum(log(Y_prob));
avg_prob = mean(Y_prob);

end%end-of-function compute_loglikelihood_MT_sequential()


%%%%
function [ LL, avg_prob, Y_prob ] = compute_loglikelihood_MT_parallel( Trees, lambda, X, Y, is_switching )

global num_batches;

K = length(Trees);
n = size(X,1);
prob = zeros(n,1);

batch_size = round(n/num_batches);
batch_size = repmat(batch_size, [1 num_batches]);

s = 1;
e = s+batch_size(1)-1;
for i = 1:num_batches
    X_temp{i} = X(s:e,:);
    Y_temp{i} = Y(s:e,:);
    
    % prep next
    s = e+1;
    e = s+batch_size(i)-1;
    if i == num_batches-1 && e ~= n
        e = n;
        batch_size(num_batches) = e-s+1;
    end
end

parfor i = 1:num_batches
    [ LL_temp(i), avg_prob_temp(i), Y_prob_temp{i} ] = compute_loglikelihood_MT_sequential( Trees, lambda, X_temp{i}, Y_temp{i}, is_switching );
end

Y_prob = [];
for i = 1:num_batches
    Y_prob = [Y_prob;Y_prob_temp{i}];
end

avg_prob = batch_size*avg_prob_temp'/n;
LL = sum(LL_temp);

end%end-of-function compute_loglikelihood_MT_parallel()

