function [ mask ] = makeCellMask ( cell_list, img_size, cell_width, label )
    if nargin < 4, label = false; end
    mask = zeros (img_size);
    if label
        for ix = 1:numel(cell_list)
            cell_mask = maskFromCellSpine (...
                cell_list(ix).refined, img_size, cell_width);
            mask = max (mask, cell_mask .* ix);
        end             
    else
        for ix = 1:numel(cell_list)
            mask = mask | maskFromCellSpine (...
                cell_list(ix).refined, img_size, cell_width);
        end
    end
end