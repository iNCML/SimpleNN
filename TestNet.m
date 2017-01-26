function err = TestNet(data, targets, net)
%
%  err = TestNet(data, targets, net)
% 
%  data: batchSize * frameSize.
%  targets: batchSize * nTargets
%  net:  the network to be trained, it should be built and initialized
%
%to: Training options:
% lr: learning rate, wdr: weight decay rate, momentum, dw{iLayer}: weight change, db{iLayer}: bias change
%err: [difference, # of errors, objective function]

tic

nLayers = length(net);
batchSize = size(data,1);
nClasses = size(targets,2);

%******* Forward activation
feat{1} = data;
feat1{1} = [feat{1} , ones(batchSize,1,'single')];
for iLayer = 1:nLayers-1
  %activation
  feat{iLayer+1} = 1./(1+exp( -feat1{iLayer} * net{iLayer}.w ) ); %batchSize * nf
  feat1{iLayer+1} = [feat{iLayer+1} , ones(batchSize,1,'single')];
end

%*** Top Layer
act = exp(feat1{nLayers} * net{nLayers}.w);
feat{nLayers+1} = act ./ repmat(sum(act,2),1,nClasses);

%*** Compute top layer errors
%compute top delta
iLayer = nLayers;
edelta = (targets - feat{iLayer+1}); %softmax with cross entropy.  batchSize * d2
err(1) = sum(sum(edelta.^2)) ./ (nClasses) / batchSize;    %  mean square error

[temp, mi1] = max(targets,[],2);
[temp, mi2] = max(feat{nLayers+1},[],2);

err(2) = sum(mi1 ~= mi2) / batchSize;    % classification error counts

err(3) = -sum(sum(targets .* log( feat{iLayer+1} + 0.1e-15 ) )) / batchSize; % cross entropy 

ttr = toc;

fprintf('TOTAL: error rate = %.2f%%, trMSE = %f, trCE = %f in %f sec \n', err(2)*100.0, err(1), err(3), ttr) ;

