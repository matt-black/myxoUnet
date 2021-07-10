function [ is_touching ] = isPixelTouchingClass (row, col, lbl_mask, conn)
%ISPIXELTOUCHINGCLASS
    % get neighborhood of cell
    if lbl_mask(row,col) == 0 
        is_touching = false;
        return;
    end
    if nargin < 4, conn = 8; end
    if conn == 4
        nhood = [row-1, col;
            row, col-1; row, col+1;
            row+1, col];
    else
        nhood = [row-1, col-1; row-1, col; row-1, col+1;
            row, col-1; row, col+1;
            row+1, col-1; row+1, col; row+1, col+1];
    end
    % get rid of invalid rows/cols
    kill = nhood(:,1) < 1 | nhood(:,1) > size (lbl_mask, 1) | ...
        nhood(:,2) < 1 | nhood(:,2) > size (lbl_mask, 2);
    nhood(kill,:) = [];
    nhood_inds = sub2ind (size (lbl_mask), nhood(:,1), nhood(:,2));
    nhood_vals = lbl_mask(nhood_inds);
    
    if any (nhood_vals > 0 & nhood_vals ~= lbl_mask(row,col))
        is_touching = true;
    else
        is_touching = false;
    end
end

