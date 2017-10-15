function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

K=ones(size(X,1),1);
% Adding X0
X=[K X];

% FOR FIRST LAYER : 

A1 = X*Theta1';
A1 = sigmoid(A1);
K=ones(size(A1,1),1);		% K is of the size of first column of A1 and is filled with ones. :")
A1 = [K A1]; 				% Adding extra parameter a(1)[0]

% For Second Layer :

A2=A1*Theta2';
A2=sigmoid(A2);
% For last Layer :

[m p] = max(A2,[],2);


% =========================================================================


end
