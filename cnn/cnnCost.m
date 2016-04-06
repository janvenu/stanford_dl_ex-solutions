function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred,weightDecay,activationType,cost_fun)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)

if nargin<11
   cost_fun = 'cross-entropy';
end

if nargin<10
   activationType = 'logistic';
end

if nargin<9
    weightDecay =1e-3;
end

if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

convDim = imageDim-filterDim+1;
outputDim = (convDim)/poolDim;

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = cnnConvolve(filterDim, numFilters, images, Wc, bc);

% outputDim x outputDim x numFilters x numImages tensor for storing subsampled activations
activationsPooled = cnnPool(poolDim, activations);

% Reshape activations into 2-d matrix, hiddenSize x numImages, for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);


%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
z = Wd*activationsPooled+repmat(bd,1,numImages);
% z = bsxfun(@minus, z, max(z));
[cost,errorsSoftmax] = cost_function_nn(labels,z,numClasses,[],cost_fun);
z = exp(z);
probs = bsxfun(@rdivide, z, sum(z));
%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = cost+ .5 * weightDecay * (sum(Wd(:) .^ 2) + sum(Wc(:) .^ 2));

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%
errorsSoftmax = errorsSoftmax/numImages;
errorsPooled = Wd' * (errorsSoftmax);
errorsPooled = reshape(errorsPooled, [], outputDim, numFilters, numImages);

errorsPooling = zeros(convDim, convDim, numFilters, numImages);
unpoolingFilter = ones(poolDim);


poolArea = poolDim ^ 2;
unpoolingFilter = unpoolingFilter / poolArea;
parfor imageNum = 1:numImages
    for filterNum = 1:numFilters
        e = errorsPooled(:, :, filterNum, imageNum);
        errorsPooling(:, :, filterNum, imageNum) = kron(e, unpoolingFilter);
    end
end

errorsConvolution = errorsPooling .* get_ce(activations,activationType);

%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

Wd_grad = (errorsSoftmax * activationsPooled');
Wd_grad = Wd_grad + weightDecay * Wd;
bd_grad = sum(errorsSoftmax, 2);

bc_grad = zeros(size(bc));
Wc_grad = zeros(size(Wc));



for filterNum = 1 : numFilters
   Wc_gradFilter = zeros(size(Wc_grad, 1), size(Wc_grad, 2));
    parfor imageNum = 1 : numImages
        e = errorsConvolution(:, :, filterNum, imageNum);
        errorsConvolution(:, :, filterNum, imageNum) = rot90(e, 2);        
        Wc_gradFilter = Wc_gradFilter + conv2(images(:, :, imageNum), errorsConvolution(:, :, filterNum, imageNum), 'valid');
    end
    Wc_grad(:, :, filterNum) = Wc_gradFilter;
end

for filterNum = 1 : numFilters
    e = errorsConvolution(:, :, filterNum, :);
    bc_grad(filterNum) = sum(e(:));
end
Wc_grad = Wc_grad + weightDecay * Wc;
%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
