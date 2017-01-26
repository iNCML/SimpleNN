function [net, err] = SimpleNN(trainData,trainTargets,MaxIte,lrate,minibatch,NNSize,Momentum,weightdecay,initRange) 
%
%   [net, err] = SimpleNN(trainData,trainTargets,MaxIte,lrate,minibatch,NNSize,Momentum,weightdecay,initRange) 
%
% Input: 
%       trainData(nSamples,inputSize)
%       trainTargets(nSamples,nClasses)
%       MaxIte : max # of epoch, 30 
%       lrate : learning rate, 0.1
%       minibatch : minibatch size, 10
%       NNSize: sizes of hidden layers, [100 100]
%       Momentum: momentum, 0.9
%       weightdecay: weight decay, 0.0
%       initRange: variance of initialization, 0.01    
%
% Output:
%       net:   trained neural network
%       err:   training errors

%############ Topology Parameters
netDesc = [100, 100] ;   % # of hidden nodes in each layer
MaxEpoch = 20;       % max # of epoch in training

% fine tuning parameters
% mbsz: minibatch size; wdr: weight decay; lr: learning rate; 
% finalMomentum: momentum; initRange: variance for weight initialization
to = struct('mbsz',10,'wdr',0.000, 'lr',0.1,'finalMomentum',0.9, 'initRange', 0.01);


if(nargin<2 | nargin>9) 
    help SimpleNN
    return
    else

 if (nargin >= 3)
   MaxEpoch = MaxIte ;
 end
 if (nargin >= 4)
   to.lr = lrate ;
 end 
 if (nargin >= 5)
   to.mbsz = minibatch ; 
 end
 if (nargin >= 6)
   netDesc = NNSize ;
 end
 if (nargin >= 7)
   to.finalMomentum = Momentum ;
 end
 if (nargin >= 8)
   to.wdr = weightdecay ;
 end
 if (nargin >= 9)
   to.initRange = initRange ; 
 end

MaxEpoch
netDesc
to

%############ Init
[nSamples, inputSize] = size(trainData);
nClasses = size(trainTargets,2);

nLayers = length(netDesc);
d1 = inputSize;
for iLayer = 1:nLayers
  net{iLayer}.w = randn(d1+1,netDesc(iLayer),'single') .* to.initRange;
  to.dw{iLayer} = zeros(d1+1,netDesc(iLayer),'single');
  d1 = netDesc(iLayer);
end
nLayers = nLayers + 1;
net{nLayers}.w = randn(d1+1,nClasses,'single') .* to.initRange ;
to.dw{nLayers} = zeros(d1+1,nClasses,'single');

%############# Train
batchSize = to.mbsz; %Size of training batches
nBatches = floor(nSamples / batchSize);
for ep = 1:MaxEpoch

  if ep == 1 %0
      to.momentum =0.0;
  else
      to.momentum = to.finalMomentum;
  end;

  tic
  totErr = 0.0;
  totLoss = 0.0;
  totMse = 0.0;
  
  randNdx = randperm(nSamples);
  data = single(trainData(randNdx,:));
  targets = single(trainTargets(randNdx,:));

  for batch = 1:nBatches
    x = data((batch-1)*batchSize+1:batch*batchSize,:);
    t = targets((batch-1)*batchSize+1:batch*batchSize,:);

    [net, err, to] = DoBackProp(x, t, net, to);

    totErr = totErr + err(2);
    totLoss = totLoss + err(3);
    totMse = totMse + err(1);
  end
  ttr = toc;
  fprintf('ep %d, trCE = %f, trFER = %.2f%%, trMSE = %f, lrate =%f, in %f sec \n', ep, (totLoss*1.0/(batchSize*nBatches)), (totErr*100.0)/(batchSize*nBatches), (totMse*100)/(batchSize*nBatches), to.lr, ttr);

  save(sprintf('/tmp/network-%d',ep),'net') ;

end;

%totalerr(1) = totMse ;
%totalerr(2) = totErr ;
%totalerr(3) = totLoss ;

fprintf(' *********************************************************\n');

%############## training error
[err] = TestNet(trainData, trainTargets, net);

save('network.mat','net');

fprintf(' *********************************************************\n');


end
