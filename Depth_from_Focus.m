clear;
close all;

% Select what dataset we will use
dataset = 2;

if dataset == 1
    addpath('PA1_dataset1_balls');
    PicNum = 24;
elseif dataset == 2
    addpath('PA1_dataset2_keyboard');
    PicNum = 32;
end

% Open source
addpath('iatRoot');
addpath('gco-v3.0\matlab');
addpath('matlab_wmf_release_v1');

% M is the height and N is the width
M = 720;
N = 1280;


%% Step1. Image Alignment

% ImageSet is the set of the gray scale pictures
% AlignImage is the set of the rgb pictures
ImageSet = zeros(PicNum, M, N);
AlignImage = zeros(PicNum, M, N, 3);

img = imread('0.jpg');
wimage = rgb2gray(img);
ImageSet(1,:,:) = wimage;
AlignImage(1,:,:,:) = img;

% Do alignment the gray scale image
% Outputs are alignmented gray scale images. 
for i = 2:PicNum
    tmp = wimage;
    img = imread(strcat(int2str(i-1), '.jpg'));
    AlignImage(i,:,:,:) = img;
    img = rgb2gray(img);

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
    %iat_plot_correspondences(img,tmp,X1(inliers,:)',X2(inliers,:)');

    [wimage, support] = iat_inverse_warping(img, ransacWarp, 'affine', 1:N, 1:M);
    %figure; imshow(tmp); figure; imshow(uint8(wimage));
    [AlignImage(i,:,:,:), support2] = iat_inverse_warping(squeeze(AlignImage(i,:,:,:)), ransacWarp, 'affine', 1:N, 1:M);
    
    ImageSet(i,:,:) = wimage;
end
figure; imshow(uint8(squeeze(ImageSet(PicNum,:,:)))); title('Feature Based Alignment');



%% Step 2 Focus Measure

IndexMap = zeros(M, N);
FM = zeros(PicNum, M, N);
Intensity = zeros(M, N);

% Calculating OTF for filtering
OTF = zeros(M,N);
sigma1 = 0.01;
sigma2 = 0.1;
var = 3.5;
for j = 1:M
    for k = 1:N
        kx = j * var*pi / M;
        ky = k * var*pi / N;
        OTF(j,k) = exp(-sigma1*(kx^2+ky^2)) - exp(-sigma2*(kx^2+ky^2));
    end
end

OTFhalf = zeros(M/2, N/2);
for j = 1:M/2
    for k = 1:N/2
        OTFhalf(j,k) = (OTF(2*j-1, 2*k-1) + OTF(2*j, 2*k-1) + OTF(2*j-1, 2*k) + OTF(2*j, 2*k)) / 4;
    end
end

OTF(1:M/2,1:N/2) = rot90(OTFhalf,2);
OTF(M/2+1:M,1:N/2) = flip(OTFhalf,2);
OTF(1:M/2,N/2+1:N) = flip(OTFhalf);
OTF(M/2+1:M,N/2+1:N) = OTFhalf;

OTF = rescale(OTF,0,255);
%figure; imshow(uint8(OTF));

% Calculating focus measure data
for i = 1:PicNum
    img = squeeze(ImageSet(i,:,:));

    % 2nd derivative of the image along the x and y axis
    for j = 2:M-1
       for k = 2:N-1 
          Intensity(j,k) = (img(j+1,k) - 2*img(j,k) + img(j-1,k)) + (img(j,k+1) - 2*img(j,k) + img(j,k-1));
       end
    end
    
    % Calculating ic
    ic = real(ifft2(fft2(Intensity.^2).*OTF));
    
    % 3 by 3 window for FM
    for j = 1:M
        for k = 1:N
            for l = -1:1
                if (j+l <= M) && (j+l >= 1) && (k+l <= N) && (k+l >= 1)
                    FM(i,j,k) = FM(i,j,k) + ic(j+l,k+l);
                end
            end
        end
    end
    %figure; imshow(uint8(rescale(squeeze(FM(i,:,:)),0,255)));
end

% Making the focus map using the max focus measure value among the frames.
for i = 1:M
    for j = 1:N
        [Max, idx] = max(FM(:,i,j));
        IndexMap(i,j) = idx;
    end
end

IndexMapCrop = IndexMap(32:M-32, 64:N-64);
figure; imshow(uint8(rescale(IndexMapCrop,0,255))); title('Focus Map');
colormap(flipud(jet)); colorbar;

% Collect edge detections from all frames
Edge = zeros(PicNum, M, N);
EdgeSum = zeros(M, N);
for i = 1:PicNum
    Edge(i,:,:) = edge(squeeze(ImageSet(i,:,:)));
    for j = 1:M
        for k = 1:N
            if Edge(i,j,k) > 0
                Edge(i,j,k) = FM(i,j,k);
            end
        end
    end
    
    EdgeSum = EdgeSum + edge(squeeze(ImageSet(i,:,:)));
end
%figure; imshow(uint8(rescale(EdgeSum,0,255)));

% Sparse depth map confident (valid) pixels
EdgeSumColor = zeros(M, N, 3);
ColorJet = flipud(jet(32));
for i = 1:M
    for j = 1:N
        if EdgeSum(i,j) > 0
            EdgeSumColor(i,j,:) = ColorJet(IndexMap(i,j),:);
        end
    end
end
EdgeSumColorCrop = EdgeSumColor(32:M-32, 64:N-64, :);
figure; imshow(uint8(rescale(EdgeSumColorCrop,0,255))); title('Edge Image');



%% Step3. Graph-cuts

% Set the data term which is minimum at the well focused frame.
mat_D = zeros(PicNum, M, N);
MaxEdge = max(max(max(Edge)));
for i = 1:M
    for j = 1:N
        if sum(Edge(:,i,j)) ~= 0
            mat_D(:,i,j) = MaxEdge - Edge(:,i,j);
        else
            mat_D(:,i,j) = 0;
        end
    end
end

% Crop and shrink dim from [PicNum, M, N] to [PicNum, M*N]
mat_D = mat_D(:, 32:M-32, 64:N-64);
mat_D = mat_D(:,:);

% Making the data term small to be used at GCO_SetDataCost
if dataset == 1
    mat_D = mat_D / (0.01*10e3);
elseif dataset == 2
    mat_D = mat_D / (1.5*10e3);
end

% Making smooth term
[M,N] = size(IndexMapCrop);
mat_S = zeros(PicNum, PicNum);
for i = 1:PicNum
    for j = 1:PicNum
        mat_S(i,j) = abs(i-j);
    end
end

% Making neighborhood for smoothing.
% The neighborhoods are four around to target pixel
idx = [M+1 : M*(N-1)]';
idx(mod(idx, M) == 0 | mod(idx, M) == 1) = [];
mat_N = sparse([idx, idx, idx, idx], [idx+1, idx-1, idx+M, idx-M], ones(size(idx,1), 4), M*N, M*N);
mat_N = mat_N';

h = GCO_Create(M*N, PicNum);
GCO_SetDataCost(h,int32(mat_D));
GCO_SetSmoothCost(h,int32(mat_S));
GCO_SetNeighbors(h,mat_N); 
GCO_Expansion(h);
Labeled_data = GCO_GetLabeling(h);
GCO_Delete(h);

GraphCuts = reshape(Labeled_data,M,N);
figure; imshow(uint8(rescale(GraphCuts,0,255))); title('Graph-cuts Result');
colormap(flipud(jet)); colorbar;



%% Step4. All in Focus Image

% Set the boundary of the crop size of pictures via the mean y of the label 
order = [];
for i = 1:PicNum
    order = [order; mean(mod(find(GraphCuts == i), M))];
end
order = sort(order);
order = [1; (order(1:end-1) + order(2:end)) / 2; M];
order = floor(order);

% Assign the crop of pictures to focus image 
FocusImage = zeros(M, N, 3);
for i = 1:PicNum
    FocusImage(order(i):order(i+1), :, :) = squeeze(AlignImage(i, 32+order(i):32+order(i+1), 64:63+N, :));
end

figure; imshow(uint8(FocusImage)); title('All in Focus Image');



%% Step5. Depth Refinement

% Set the parameter for the weighted median filter
num_disp = PicNum;

dispMapInput  = GraphCuts;
imgGuide = FocusImage;

r = ceil(max(size(imgGuide, 1), size(imgGuide, 2)) / 40);
eps = 0.01^2;

dispMapOutput = weighted_median_filter(dispMapInput, imgGuide, 1:num_disp, r, eps);
dispMapOutput = medfilt2(dispMapOutput,[3,3]);

figure; imshow(uint8(rescale(dispMapOutput,0,255))); title('Depth Refinement');
colormap(flipud(jet));

