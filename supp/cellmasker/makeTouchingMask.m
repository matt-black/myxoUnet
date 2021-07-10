function [ mask ] = makeTouchingMask ( lbl_cell_mask, conn )
    if nargin == 1, conn = 8; end
    [cols, rows] = meshgrid (1:size(lbl_cell_mask, 2), ...
                             1:size(lbl_cell_mask, 1));
    function [ yn ] = istouch ( row, col )
        yn = isPixelTouchingClass (row, col, lbl_cell_mask, conn);
    end
    mask = arrayfun (@istouch, rows, cols);
end