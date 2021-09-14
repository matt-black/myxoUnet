function [ cell_dist, neig_dist, lbl_mask, bord_mask ] = generateDistImgs ( ...
    cell_list, img_size, cell_width, bord_strel, neig_strel, norm_dist )
%GENERATEMASKS 
    if nargin < 6, norm_dist = true;
        if nargin < 5, neig_strel = []; 
            if nargin < 4, bord_strel = strel ('square', 3); end
        end
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
    % touching mask
    [cols, rows] = meshgrid (1:img_size(2), 1:img_size(1));
    touch_mask = arrayfun (...
        @(r, c) isPixelTouchingClass (r, c, lbl_mask, 4), rows, cols);
    touch_mask = bwmorph (touch_mask, 'skel');

    cell_mask = (lbl_mask > 0) & not (touch_mask);
    lbl_mask = bwlabel (cell_mask, 4);
    
    [cls3mask, cd_prelim] = augmentBorderClass (cell_mask, bord_strel);
    bord_mask = cls3mask == 2;
    
    % generate cell/neighbor distances for each cell
    idz = unique (lbl_mask(:));
    idz = idz(idz>0);
    cell_dist = zeros (img_size);
    neig_dist = zeros (img_size);
    for idx = 1:numel(idz)
        id = idz(idx);
        % find this cell
        this_cell = lbl_mask == id;
        [r, c] = find (this_cell);
        cell_ind = sub2ind (img_size, r, c);
        % normalize cell distance (above) to [0,1] for cell
        dvalz = cd_prelim(cell_ind);
        if norm_dist
            dvalz = dvalz ./ max (dvalz);
        end
        cell_dist(cell_ind) = dvalz;

        % compute neighbor distances (& assign)
        this_neig = lbl_mask > 0 & lbl_mask ~= id;
        this_neig_dist = bwdist (this_neig, 'euclidean');
        % normalize to [0,1] and assign
        nvalz = this_neig_dist(cell_ind);
        nvalz = 1 - normalizeVector (nvalz);
        neig_dist(cell_ind) = nvalz;
    end
    % closing and scaling of neighbor distances
    if not (isempty (neig_strel))
        neig_dist = imclose (neig_dist, neig_strel);
    end
    neig_dist = neig_dist .^ 3;         % scale to power 3
end

function [ norm_vec ] = normalizeVector ( vec )
% NORMALIZEVECTOR normalize values of vector to range [0,1]
    norm_vec = (vec - min(vec(:))) ./ range (vec);
end