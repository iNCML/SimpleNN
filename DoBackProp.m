function [net, err, to] = DoBackProp(data, targets, net, to)
%data: batchSize * frameSize.
%targets: batchSize * nTargets
%net:  the network to be trained, it should be built and initialized
%to: Training options:
% lr: learning rate, wdr: weight decay rate, momentum, dw{iLayer}: weight change, db{iLayer}: bias change
%err: [difference, # of errors, objective function]

nLayers = length(net);
batchSize = size(data,1);
nClasses = size(targets,2);

%******* Forward activation
feat{1} = data;
feat1{1} = [feat{1} , ones(batchSize,1,'single')];  % append 1's for biases 
for iLayer = 1:nLayers-1
  %activation
  feat{iLayer+1} = 1./(1+exp( -feat1{iLayer} * net{iLayer}.w ) ); %batchSize * nf
  feat1{iLayer+1} = [feat{iLayer+1} , ones(batchSize,1,'single')];
end

%*** Top Layer using soft-max
act = exp(feat1{nLayers} * net{nLayers}.w);
feat{nLayers+1} = act ./ repmat(sum(act,2),1,nClasses);

%*** Compute top layer errors
%compute top delta
iLayer = nLayers;
edelta = (targets - feat{iLayer+1}); %softmax with cross entropy.  batchSize * d2
err(1) = sum(sum(edelta.^2)) ./ (nClasses);   % mean squre error

[temp, mi1] = max(targets,[],2);
[temp, mi2] = max(feat{nLayers+1},[],2);

err(2) = sum(mi1 ~= mi2);    % error counts

err(3) = -sum(sum(targets .* log( feat{iLayer+1} + 0.1e-15 ) ));  % cross entropy 

%Update top weights
edelta = edelta / batchSize;
delta2 = edelta * net{iLayer}.w(1:end-1,:)';
to.dw{iLayer} = to.lr .* (feat1{iLayer}' * edelta) + to.momentum .* to.dw{iLayer};
net{iLayer}.w = net{iLayer}.w + to.dw{iLayer} - (net{iLayer}.w) * to.wdr;   % plus weight decay

%******* Backpropagation

for iLayer = nLayers-1:-1:1

  %signal through activation function
  edelta = delta2 .* feat{iLayer+1} .* (1-feat{iLayer+1}); % batchSize * (nKernels{iLayer}*nOutBands{iLayer})

  %signal through weights
  delta2 = edelta * net{iLayer}.w(1:end-1,:)'; %batchSize*inputSize

  %weight change
  to.dw{iLayer} = to.lr .* ( transpose(feat1{iLayer}) * edelta) + to.momentum .* to.dw{iLayer};

  %*** Update weights
  net{iLayer}.w = net{iLayer}.w + to.dw{iLayer} - (net{iLayer}.w) * (to.wdr * batchSize);

end

