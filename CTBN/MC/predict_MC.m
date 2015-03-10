function [Y_pred, Y_log_prob] = predict_MC(MC_Model, X_test, Y_test)

% init params
SA=1;NAIVE=2;
if size(Y_test,2) > 6
    MAP_METHOD = SA;
else
    MAP_METHOD = NAIVE;
end

if MAP_METHOD == SA
    [Y_pred, Y_log_prob] = MAP_prediction_MT_SA(MC_Model.trees, MC_Model.lambda, X_test, Y_test, MC_Model.is_switching);
elseif MAP_METHOD == NAIVE
    [Y_pred, Y_log_prob] = MAP_prediction_MT_naive(MC_Model.trees, MC_Model.lambda, X_test, Y_test, MC_Model.is_switching);
end

end%end-of-function predict_MC()