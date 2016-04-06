function a = activation_function_nn(z,ei)

if nargin <2
    ei = 'logistic';
end
switch ei
    case 'logistic'
        a = sigmoid(z);
    case 'tanh'
        a = tanh(z);
    case 'rect_lin'
        a = max(0,z);
end