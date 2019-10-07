clear;
close all;

addpath('PA1_dataset2_keyboard');
addpath('gco-v3.0\matlab');

%% Step1. Image Alignment
tmp = imread('1.jpg');
img = imread('2.jpg');

[d1, l1]=iat_surf(img);
[d2, l2]=iat_surf(tmp);
[map, matches, imgInd, tmpInd]=iat_match_features_mex(d1,d2,.7);
X1 = l1(imgInd,1:2);
X2 = l2(tmpInd,1:2);
%iat_plot_correspondences(img, tmp, X1', X2');

% With Ransac
X1h = iat_homogeneous_coords (X1');
X2h = iat_homogeneous_coords (X2');
[inliers, ransacWarp]=iat_ransac( X2h, X1h,'affine','tol',.05, 'maxInvalidCount', 10);
iat_plot_correspondences(img,tmp,X1(inliers,:)',X2(inliers,:)');

[M,N,L] = size(tmp);
[wimage, support] = iat_inverse_warping(img, ransacWarp, 'affine', 1:N, 1:M);
figure; imshow(tmp); figure; imshow(uint8(wimage));
