function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
J_1 = 0;
for i=1:m
  h_theta_x_i = sigmoid(X(i,:)*theta);
  J_1 += (-y(i)*log(h_theta_x_i)-(1-y(i))*log(1-h_theta_x_i));
  for j=1:size(theta,1)
    grad(j) += (h_theta_x_i-y(i))*X(i,j);
  endfor
endfor

J_2 = 0;
for j=2:size(theta,1)
  theta_j_2 = theta(j)*theta(j);
  J_2 += theta_j_2;
  grad(j) += lambda*theta(j);
endfor


J = J_1/m+J_2*lambda/(2*m);
grad = grad/m;





% =============================================================

end
