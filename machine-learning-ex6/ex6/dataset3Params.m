function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
min_prediction_error = 999999999;
for i = 1:length(Cs)
    for j = 1:length(sigmas)
        C_now = Cs(i);
        sigma_now = sigmas(j);
        model = svmTrain(X, y, C_now, @(x1, x2) gaussianKernel(x1, x2, sigma_now));
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));
        display(prediction_error);
        if prediction_error < min_prediction_error
            C = C_now;
            sigma = sigma_now;
            min_prediction_error = prediction_error;
        end
    end
end

display('Final C, sigma and min_prediction_error:');
display(C);
display(sigma);
display(min_prediction_error);

% =========================================================================

end
