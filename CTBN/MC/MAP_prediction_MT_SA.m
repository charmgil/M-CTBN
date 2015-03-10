function [Y_pred, Y_log_prob]= MAP_prediction_MT_SA( Trees, lambda, X, Y, is_switching, max_iter )

if ~exist('max_iter','var')
    max_iter = 150;
end

is_verbose = false;
K = length(Trees);
n = size(X,1);
Y_log_prob = zeros(n, 1);

for i = 1:n
    if is_verbose
        fprintf( '%d/%d\n', i, n);
    elseif mod(i,10) == 1
        fprintf( '.', i, n);
    end
    
    x = X(i,:);
    y = Y(i,:);
    
    Tx = [];
    for k = 1:K
        Tx{k} = compute_log_potentials( Trees{k}, x, is_switching );
    end
    
    % s=pred, s_new=pred_new; e=prob, e_new=prob_new
    % opt1: choose best lambda
%     [~,k0] = max(lambda);
%     s0 = MAP_prediction( Trees{k0}, x, is_switching );
    
    % opt2: choose best prob
    for k=1:K
        s0_pred{k} = MAP_prediction( Trees{k}, x, y, is_switching );
        s0_prob(k) = compute_energy( Tx, lambda, s0_pred{k} );
    end
    [~,s0_best] = min(s0_prob);
    s0 = s0_pred{s0_best};

    % opt3: dummy (all-0)
    %s0 = zeros( 1, length(Trees{k}) );
    
    ObjFn = @(s) compute_energy( Tx, lambda, s );
        options = saoptimset('DataType', 'custom');
        options = saoptimset(options, 'TemperatureFcn', @temperatureexp, 'InitialTemperature', 100);
        options = saoptimset(options, 'AnnealingFcn', @neighbor);
        options = saoptimset(options, 'AcceptanceFcn', @acceptancesa);
        
        options = saoptimset(options, 'MaxIter', max_iter);
        options = saoptimset(options, 'ReannealInterval', 50);
        
        if is_verbose
            options = saoptimset(options, 'Display', 'iter', 'DisplayInterval', 100);
        else
            options = saoptimset(options, 'Display', 'off');
        end
        
    [s_best, fval, exitFlag, output] = simulannealbnd(ObjFn, s0, [], [], options);
    
    Y_pred(i,:) = s_best;
    Y_log_prob(i,1) = log(-compute_energy( Tx, lambda, y ));
end
end %end-of-function MAP_prediction_MT_SA()


function e = compute_energy( Tx, lambda, s )
%e = -compute_loglikelihood_MT( Trees, lambda, x, s, is_switching );
for k = 1:length(Tx)
    tmp_prob(k) = exp(evaluate_probability( Tx{k}, s ));
end
e = -dot(tmp_prob, lambda);

end %end-of-function compute_prob_MT()


function s_new = neighbor(optimvalues,problem)

s = optimvalues.x;
d = length(s);

s_new = s;
i = randi(d,1,1);
s_new(i) = -(s_new(i)-1);   %flip
end %end-of-function neighbor()



% function s_new = neighbor_n(optimvalues,problem)
% 
% s = optimvalues.x;
% d = length(s);
% 
% n_flips = round(d/2 * mean(optimvalues.temperature)/100);
% if n_flips == 0, n_flips = 1; end
% 
% s_new = s;
% j = 0;
% is_flipped = zeros(1,d);
% while j < n_flips
%     i = randi(d,1,1);
%     if ~is_flipped(i)
%         s_new(i) = -(s_new(i)-1);   %flip
%         is_flipped(i) = 1;
%         j = j + 1;
%     end
% end
% end %end-of-function neighbor()


