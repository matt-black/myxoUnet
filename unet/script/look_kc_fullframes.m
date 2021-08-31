clear, clc, close all
%% load data
load (fullfile (pwd, 'analyze_kc_fullframes.mat'))
I = uint16 (I);  % images are actually uint16
P = P == 1;  % these are cell masks

%% example
ix = 1;

img = squeeze (I(ix,:,:));
msk = squeeze (P(ix,:,:));

figure(1), clf
imshowpair (img, msk, 'montage')

%%

figure(1)
imshow (img), hold on
bnd = bwboundaries (msk);
for bi = 1:numel(bnd)
    b = bnd{bi};
    if size (b,1) < 20, continue, end
    plot (b(:,2), b(:,1))
end, hold off
