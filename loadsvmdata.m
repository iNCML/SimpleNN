function [X, Y, YY] = loadsvmdata(file)
% Load LIBSVM data format 
%
  %  Usage: [X, Y, YY] = loadsvmdata(file)
%
%  Author: Hui Jiang

  fid = fopen(file) ;
  
  line=1 ;
  while  ~feof(fid)  
    tline = fgets(fid) ;

    [y, count, err, nextindex] = sscanf(tline, '%d', 1 ) ;
    N = size(tline,2) ;
    A = sscanf(tline(nextindex:N), '%d:%f') ;
%    X(1,line) = 1.0 ;    % for bias
    NN = size(A,1) ;
    for i=1:2:NN,
	    X(A(i),line) = A(i+1) ;
    end
    Y(line) = y ;
    line = line + 1 ;
  end 

  X(size(X,1)+1,:) = ones(size(X,2),1) ; % add for bias

  Y = Y' ;
  y1 = unique(Y) ;   % expand label to matrix
  n = size(y1,1) ;
  for i=n:-1:1
   YY(i,:) = (Y == y1(i)) ;
  end
  
  fclose(fid) ;
