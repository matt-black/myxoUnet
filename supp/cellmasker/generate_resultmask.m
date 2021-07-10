clear, clc, close all
%%
addpath (fullfile (pwd, 'supp'));
load ('example_result.mat')

% filter down to approved cells, only
keep = arrayfun (@(c) c.approved, cellList);
cellList = cellList(keep);

% make labeled cell mask
img_size = size (Image);
lbl_mask = zeros (img_size);
mskfn = @(spn) maskFromCellSpine (spn, img_size, 4);
for ii = 1:numel(cellList)
    lbl_mask = max (lbl_mask, mskfn (cellList(ii).refined) .* ii);
end

% compute "gap" mask
str = strel ('disk', 5);
bh_mask = imbothat (lbl_mask>0, str);
gap_mask = bh_mask & (lbl_mask == 0);

% touching mask
[cols, rows] = meshgrid (1:size(Image,2), 1:size(Image,1));
touch_mask = arrayfun (...
    @(r, c) isPixelTouchingClass (r, c, lbl_mask, 8), rows, cols);
imshow (touch_mask)

semantic = max (max ((lbl_mask>0) .* 1, touch_mask .* 2), gap_mask .* 3);

imshow (label2rgb (semantic, 'jet', 'k', 'noshuffle'))
