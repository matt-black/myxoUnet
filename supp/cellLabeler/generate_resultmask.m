clear, clc, close all
%%
addpath (fullfile (pwd, 'supp'));
load ('may20_frame500.mat')
show = @(L) imshow (label2rgb (L, 'jet', 'k', 'shuffle'));
% filter down to approved cells, only
keep = arrayfun (@(c) c.approved, cellList);
cellList = cellList(keep);

% make labeled cell mask
img_size = size (Image);
lbl_mask = zeros (img_size);
mskfn = @(spn) maskFromCellSpine (spn, img_size, 4);
for ii = 1:numel(cellList)
    this_cell = maskFromCellSpine (cellList(ii).refined, img_size, 4.5);
    lbl_mask = max (lbl_mask, this_cell .* ii);
end
show (lbl_mask)

% touching mask
[cols, rows] = meshgrid (1:size(Image,2), 1:size(Image,1));
touch_mask = arrayfun (...
    @(r, c) isPixelTouchingClass (r, c, lbl_mask, 8), rows, cols);
touch_mask = bwmorph (touch_mask, 'skel');

% unlabeled cell mask
cell_mask = (lbl_mask > 0) & not (touch_mask);
% augmented cell mask
aug_se = strel ('square', 2);
cls3mask = augmentTouchingClass (cell_mask, aug_se);

% gap mask
bh_se = strel ('disk', 5);
I = imbothat (cell_mask, bh_se);
cls4mask = cls3mask;
for r = 1:size(I, 1)
    for c = 1:size(I, 2)
        if I(r,c) > 0 && cls3mask(r,c) == 0
            cls4mask(r,c) = 3;
        end
    end
end