function [ aug_mask ] = augmentTouchingClass ( cell_mask, se )
%AUGMENTTOUCHINGCLASS Algorithm 1 of arxiv:1802.07465v1
    gp = imclose (cell_mask, se);
    gp = gp - cell_mask;
    gp = imdilate (gp, se);
    aug_mask = cell_mask + (max (cell_mask)+1) .* gp;
    aug_mask(aug_mask>1) = 2;
    aug_mask(cell_mask==1) = 1;
end