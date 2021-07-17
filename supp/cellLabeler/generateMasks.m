function [ varargout ] = generateMasks ( cell_list, img_size, cell_width, ...
    aug_se, bothat_se )
%GENERATEMASKS 
    
    % cell labels
    lbl_mask = zeros (img_size);
    for ii = 1:numel(cellList)
        this_cell = maskFromCellSpine (cell_list(ii).refined, ...
            img_size, cell_width);
        lbl_mask = max (lbl_mask, this_cell .* ii);
    end
    varargout{1} = lbl_mask;
    if nargout == 1, return; end
    
    % touching mask
    [cols, rows] = meshgrid (1:size(Image,2), 1:size(Image,1));
    touch_mask = arrayfun (...
        @(r, c) isPixelTouchingClass (r, c, lbl_mask, 8), rows, cols);
    touch_mask = bwmorph (touch_mask, 'skel');

    cell_mask = (lbl_mask > 0) & not (touch_mask);
    
    % augmented cell mask
    cls3mask = augmentTouchingClass (cell_mask, aug_se);
    varargout{2} = cls3mask;
    if nargout == 2, return; end
    
    % gap mask
    I = imbothat (cell_mask, bothat_se);
    cls4mask = cls3mask;
    for r = 1:size(I, 1)
        for c = 1:size(I, 2)
            if I(r,c) > 0 && cls3mask(r,c) == 0
                cls4mask(r,c) = 3;
            end
        end
    end
    varargout{3} = cls4mask;
    if nargout == 3, return; end
    
    varargout{4} = cell_mask;
end

