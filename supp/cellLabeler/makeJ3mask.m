function [ mask ] = makeJ3mask ( cell_list, img_size, cell_width, conn )
%MAKEJ3CELLMASK
    narginchk (2, 4);
    if nargin < 4, conn = 8;
        if nargin < 3, cell_width = 3;
        end
    end 
    % generate masks for each class
    cell_lbl_mask = makeCellMask (cell_list, img_size, cell_width, true);
    touch_mask = makeTouchingMask (cell_lbl_mask, conn);
    
    % make output mask
    mask = max (cell_lbl_mask > 0, touch_mask .* 2);
end