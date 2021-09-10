function [ cell_dist, neig_dist, lbl_mask ] = generateDistImgs ( cell_list, img_size, cell_width, neig_strel )
%GENERATEMASKS 
    if nargin < 4, neig_strel = []; end
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

        % compute distance transform & assign
        this_cell_dist = bwdist (~this_cell, 'euclidean');
        cell_dist(cell_ind) = normalizeVector (this_cell_dist(cell_ind));
        
        % compute neighbor distances (& assign)
        this_neig = lbl_mask > 0 & lbl_mask ~= id;
        this_neig_dist = bwdist (this_neig, 'euclidean');
        neig_dist_vals = normalizeVector (this_neig_dist(cell_ind));
        neig_dist(cell_ind) = 1 - neig_dist_vals;
    end
    % HACK: fix this
    neig_dist(isnan(neig_dist(:))) = 0;
    cell_dist(isnan(cell_dist(:))) = 0;
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