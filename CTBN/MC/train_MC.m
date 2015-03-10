% Mixtures-of-CTBNs [Hong, Batal, Hauskrecht 2014]
function Model = train_MC(X_train, Y_train, K)

% init MC Params
n_iter = 15;        % decides the number of EM steps for the 
n_iter_last = 30;   % decides the number of EM steps for the
is_switching = true; % This decides how to represent the posterior class distribution P(y|x)
                     % If true, it follows the method described as in [Batal, Hong, Hauskrecht 2013] (switching);
                     % if false, P(y_i|x,pa(y_i)) is directly modeled as in Classifier Chains [Read et al. 2009]

Base_trees = [];
Model_tmp = cell(1,K);
Y_train_log_prob = cell(1,K);

% train an initial CTBN
W_train{1} = ones(size(X_train,1), 1)/size(X_train,1);
Base_trees{1} = learn_weighted_structure(X_train, Y_train, W_train{1}, [], is_switching);
Model_tmp{1}.trees = Base_trees;
Model_tmp{1}.lambda = 1;

% train 2:K CTBNs
for k = 2:K
    [~,~,Y_train_log_prob{k-1}] = compute_loglikelihood_MT(Model_tmp{k-1}.trees, Model_tmp{k-1}.lambda, X_train, Y_train, is_switching);
    W_train{k} = 1 - exp(Y_train_log_prob{k-1});
    W_train{k} = W_train{k} / sum(W_train{k});

    % 'learn_weighted_structure' takes previous trees as argument to check existance
    Base_trees{k} = learn_weighted_structure( X_train, Y_train, W_train{k}, Base_trees, is_switching );

    if k ~= K
        [Model_tmp{k}.trees, Model_tmp{k}.lambda] = learn_output_tree_MT( X_train, Y_train, Base_trees, is_switching, [], [], n_iter );
    else
        [Model_tmp{k}.trees, Model_tmp{k}.lambda] = learn_output_tree_MT( X_train, Y_train, Base_trees, is_switching, [], [], n_iter_last );
    end
end

Model.trees = Model_tmp{K}.trees;
Model.lambda = Model_tmp{K}.lambda;
Model.is_switching = is_switching;

end%end-of-function train_MC()