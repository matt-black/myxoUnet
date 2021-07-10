function [ mask ] = makeGapMask ( cell_mask, bothat_strel )
%MAKEGAPMASK
    bh = imbothat (cell_mask > 0, bothat_strel);
    mask = bh & (cell_mask == 0);
end