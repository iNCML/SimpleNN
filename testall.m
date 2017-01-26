function [err]=testall(testData, testTargets, N)
%
for i=1:N
 load(sprintf('/tmp/network-%d',i)) ;

 fprintf('Model %d  :',i) ;
 [err] = TestNet(testData, testTargets, net); 

end
 
