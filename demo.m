% main_postcard.m
%
% Author: Daniele Perrone, perrone@iam.unibe.ch
% Copyright (C) 2014 Daniele Perrone. All rights reserved.

clear
addpath('lib/');
%% -----------------------------------------------------------------------%
%                                  Input                                  %
% ------------------------------------------------------------------------%
g = imread('input.png');
%% -----------------------------------------------------------------------%
%                                Parameters                               %
% ------------------------------------------------------------------------%
% kernel size
MK = 19;
NK = 19;
% regularization parameter
lambda = 0.016;
% Lp norm
params.P = 2; 
params.gammaCorrection = false;
params.gamma = 0;
% number of iterations in each pyramid level
params.iters = 800;
params.visualize = 1;
% increase kernel size at each pyramid level, the larger, the fewer level
params.kernelSizeMultiplier = 1.2;
%% -----------------------------------------------------------------------%
%                                  Deblur                                 %
% ------------------------------------------------------------------------%
tic
[u, k] = deblur(g,MK,NK,lambda, params);
toc
u(u<0) = 0;
u(u>1) = 1;
imwrite(u,'out.png');
imwrite(imresize(k./max(k(:)),5,'nearest'),'kernel.png');

