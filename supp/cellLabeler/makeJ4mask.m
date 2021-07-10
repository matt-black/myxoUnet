function [ mask ] = makeJ4mask ( cell_list, bothat_strel, img_size, ...
                                 cell_width, conn )
%MAKEJ4CELLMASK
    narginchk (3, 5);
    if nargin < 5, conn = 8;
        if nargin < 4, cell_width = 3;
        end
    end 
    % generate masks for each class
    cell_lbl_mask = makeCellMask (cell_list, img_size, cell_width, true);
    gap_mask = makeGapMask (cell_lbl_mask>0, bothat_strel);
    touch_mask = makeTouchingMask (cell_lbl_mask, conn);
    % make output mask
    mask = max (cell_lbl_mask > 0, ...
                max (gap_mask .* 3, touch_mask .* 2));
end