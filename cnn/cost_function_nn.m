function [cost,temp_delta] = cost_function_nn(labels,z,num_classes,theta,cost_fun)

if nargin <5
    cost_fun = 'cross-entropy';
end

if nargin<3
    num_classes = length(unique(labels));
end

if (nargin<4)||(isempty(theta))
    theta = ones(1,num_classes);
end


numSamples = length(labels);


groundTruth = sparse(labels,1:numSamples,1,num_classes,numSamples);
switch cost_fun
    case 'cross-entropy'
        for k =1:num_classes
            z(k,:) = z(k,:).*theta(k);
        end
        z = bsxfun(@rdivide,exp(z),sum(exp(z)));
        temp_z = log(z);
        temp = sub2ind(size(z), labels', 1:numSamples);
        cost = -1*sum(temp_z(temp))/numSamples;
        temp_delta = -1*(groundTruth-z);
    case 'mean-error-sq'
        temp_z = bsxfun(@rdivide,exp(z),sum(exp(z)));
        cost = sum(((temp_z-groundTruth).^2))/numSamples;
        temp_delta = -1*(groundTruth-temp_z);
end