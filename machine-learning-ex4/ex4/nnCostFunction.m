function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% size(Theta1)             
% size(Theta2)
             
% Setup some useful variables
m = size(X, 1);

X = [ones(m, 1) X];
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

A2 = [ones(m,1) sigmoid(X * Theta1')];
A3 = sigmoid(A2 * Theta2');
y_matrix = zeros(size(y, 1), num_labels);
for i = 1:m
    y_matrix(i, y(i)) = 1;
end
J = 1 / m * sum(sum(-y_matrix .* log(A3) - (1 - y_matrix) .* log(1 - A3)));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% for i = 1:m
%     xi = X(i, :);
%     yi = y(i);
%     z2 = xi * Theta1';
%     a2 = [1 sigmoid(z2)];
%     a3 = sigmoid(a2 * Theta2');
%     y_v = zeros(1, num_labels);
%     y_v(yi) = 1;
%     delta_3 = a3 - y_v;
%     delta_2 = delta_3 * Theta2 .* sigmoidGradient([1 z2]);
%     delta_2 = delta_2(2:end);
%     Theta1_grad = Theta1_grad +  delta_2' * xi;
%     Theta2_grad = Theta2_grad + delta_3' * a2;
% end


Z2 = X * Theta1';
delta_3 = A3 - y_matrix;
delta_2 = delta_3 * Theta2 .* sigmoidGradient([ones(m,1) Z2]);
delta_2 = delta_2(:, 2:end);
Theta1_grad = delta_2' * X;
Theta2_grad = delta_3' * A2;

Theta1_grad = Theta1_grad / m;
regu1 = lambda / m * Theta1;
regu1(:, 1) = 0;
Theta1_grad = Theta1_grad + regu1;
Theta2_grad = Theta2_grad / m;
regu2 = lambda / m * Theta2;
regu2(:, 1) = 0;
Theta2_grad = Theta2_grad + regu2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% regu = lambda / m / 2 * sum(nn_params .^ 2);
regu = lambda / 2 / m * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));
J = J + regu;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end