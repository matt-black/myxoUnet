function [ rect ] = cropBoxFromCellEnds(xse, yse, pad, square)
%CROPBOXFROMCELLENDS
    if nargin < 4
        square = true;
    end
    % figure out length and angle of cell
    cell_ang = atan2 (diff (yse), diff (xse));
    % add onto end
    ext1 = [xse(1), yse(1)] + pad .* ...
        ([cos(cell_ang+pi),sin(cell_ang+pi)]);
    ext2 = [xse(2), yse(2)] + pad .* ...
        ([cos(cell_ang),sin(cell_ang)]);
    % extend out perpendicular
    x_mu = mean (xse); y_mu = mean (yse);
    ext3 = [x_mu, y_mu] + ...
        (pad .* [cos(cell_ang+pi/2),sin(cell_ang+pi/2)]);
    ext4 = [x_mu, y_mu] + ...
        (pad .* [cos(cell_ang-pi/2),sin(cell_ang-pi/2)]);
    % figure out bounding box
    candx = [ext1(1), ext2(1), ext3(1), ext4(1)];
    candy = [ext1(2), ext2(2), ext3(2), ext4(2)];
    if (square)
        dim = max (range(candx), range(candy));
        rect = [min(candx), min(candy), dim, dim];
    else
        rect = [min(candx), min(candy), range(candx), range(candy)];
    end
end