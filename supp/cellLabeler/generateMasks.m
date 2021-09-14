function [ varargout ] = generateMasks ( cell_list, img_size, ...
    cell_width, aug_se, bothat_se, cell_edge_type )
%GENERATEMASKS 
    % make sure edge type is okay (either border or touching)
    valid_type = cellfun (@(a) strcmp (a, cell_edge_type), ...
        {'border','touch','touching'});
    valid_type = any (valid_type);
    if ~valid_type
        error ('invalid cell_edge_type');
    end
    % cell labels
    lbl_mask = zeros (img_size);
    for ii = 1:numel(cell_list)
        if isfield (cell_list(ii), 'mask')
            this_cell = cell_list(ii).mask;
            % FIXME/HACK: why is this necessary for some masks?
            this_cell = this_cell(1:img_size(1), 1:img_size(2));
        else
            try
                this_cell = maskFromCellSpine (cell_list(ii).refined, ...
                                               img_size, cell_width);
            catch
                fprintf ('error while making cell %d\n', ii)
                continue
            end
        end
        % zeros in mask get colored, keep this distance
        [r, c] = find (this_cell & not (lbl_mask));
        indz = sub2ind (img_size, r, c);
        lbl_mask(indz) = ii;
    end
    varargout{1} = lbl_mask;
    if nargout == 1, return; end
    
    % touching mask
    [cols, rows] = meshgrid (1:img_size(2), 1:img_size(1));
    touch_mask = arrayfun (...
        @(r, c) isPixelTouchingClass (r, c, lbl_mask, 8), rows, cols);
    touch_mask = bwmorph (touch_mask, 'skel');

    cell_mask = (lbl_mask > 0) & not (touch_mask);
    
    % augmented cell mask
    if startsWith (cell_edge_type, 't')
        cls3mask = augmentTouchingClass (cell_mask, aug_se);
        if nargout == 5, varargout{5} = nan; end
    else
        [cls3mask, cell_dist] = augmentBorderClass (cell_mask, aug_se);
        if nargout == 5, varargout{5} = cell_dist; end
    end
    varargout{2} = uint8 (cls3mask);
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
    varargout{3} = uint8 (cls4mask);
    if nargout == 3, return; end
    
    varargout{4} = cell_mask;
end

