function [Y_pred, Y_log_prob] = MAP_prediction_MT_naive( Trees, lambda, X, Y, is_switching )

if ~exist('is_switching','var')
    is_switching = false;
end

K = length(Trees);
d = length(Trees{1});
n = size(X,1);
Y_log_prob = zeros(n, 1);

Y_comb = generate_all_bin_combinations(d);

for i=1:n
	if mod(i,10) == 1
		fprintf( '.' );
    end
    
    x=X(i,:);
    
    Y_comb_prob = zeros( size(Y_comb,1), 1 );
    
    Tx = [];
    for k = 1:K
        Tx{k} = compute_log_potentials( Trees{k}, x, is_switching );
    end
    
    for yi = 1:size(Y_comb,1)
        y = Y_comb(yi,:);
        %[~, Y_comb_prob(yi)]= compute_loglikelihood_MT( Tx, lambda, x, y, is_switching );
        for k = 1:K
            tmp_prob(k) = exp(evaluate_probability( Tx{k}, y ));
        end
        Y_comb_prob(yi) = dot(tmp_prob, lambda);
        
    end
    
    [~,Y_map_i] = max(Y_comb_prob);
    Y_pred(i,:) = Y_comb(Y_map_i,:);
    
    % Y_log_prob
    for k = 1:K
    	tmp_prob(k) = exp(evaluate_probability( Tx{k}, Y(i,:) ));
    end
    Y_log_prob(i,1) = log(dot(tmp_prob, lambda));    
end

fprintf( '\n' );

end