
fprintf 'loading training and test data ...\n' ;

% load training data

[X,Y,YY] = loadsvmdata('/cs/home/hj/scratch/hj/Data4ML/MNIST/mnist.scale') ;
XX=transpose(X(1:784,:)) ;
YY=YY' ;

[TX,TY,TYY] = loadsvmdata('/cs/home/hj/scratch/hj/Data4ML/MNIST/mnist.scale.test') ;
TXX=transpose(TX(1:784,:)) ;
TYY=TYY' ;

%fprintf 'training NN ...\n' ;

%[net, err] = SimpleNN(XX, YY) ;

%fprintf 'evaluating NN ...\n' ;

%[err] = TestNet(TXX,TYY, net);

%[err] = testall(TXX,TYY,20) ;
