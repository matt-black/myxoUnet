function [ cell_prob, lbl_mask, bord_mask ] = generateProbImgs ( ...
    cell_list, img_size, cell_width, bord_strel, ...
    exp_bound_prob, exp_outside_prob )
%GENERATEMASKS
    if nargin < 6, exp_outside_prob = -4;
        if nargin < 5, exp_bound_prob = -1/2; 
            if nargin < 4, bord_strel = strel ('square', 3); end
        end
    end
    
    % cell labels => cell mask
    lbl_mask = zeros (img_size);
    cell_prob = zeros (img_size);
    for ii = 1:numel(cell_list)
        if isfield (cell_list(ii), 'mask')
            this_cell = cell_list(ii).mask == 1;
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
        % get rid of huge overlapping cells
        if sum (lbl_mask(indz) > 0) > length(indz)*(4/5)
            continue
        end
        if isempty (indz), continue, end
        lbl_mask(indz) = ii;
        % compute mesh
        try
            bnd = bwtraceboundary (this_cell, [r(1) c(1)], 'w');
        catch
            bnd = bwtraceboundary (this_cell, [r(1) c(1)], 'n');
        end
        bnds = smoothContour (bnd, ceil (size(bnd,1)/4));
        mesh = model2MeshForRefine (makeccw (bnds), ...
            0.25, 1e-3, 4);
        % formulate interpolant
        [rr, cc] = meshgrid (max(min(r)-2,1):1:min(max(r)+2,img_size(1)), ...
            max(min(c)-2,1):1:min(max(c)+2,img_size(2)));
        pts = unique ([rr(:), cc(:)], 'rows');
        if all (mesh(:) == 0)  
            % if mesh computation fails, fallback to skeletonized-centerline
            [cr, cc] = find (bwskel (this_cell));
            cent_line = [cr, cc];
            % formulate interpolant:
            % points on centerline get Pr=1 that we're on a cell
            % points on border get Pr=exp(-1/2) that we're on a cell
            Fr = vertcat (cent_line(4:end-4,1), bnds(:,1));
            Fc = vertcat (cent_line(4:end-4,2), bnds(:,2));
            [in, on] = inpolygon (pts(:,1), pts(:,2), bnds(:,1), bnds(:,2));
            out = not (or (in, on));
            Fr = vertcat (Fr, pts(out,1));
            Fc = vertcat (Fc, pts(out,2));
            Fv = vertcat (zeros (size (cent_line(4:end-4,:),1), 1), ...
                ones (size (bnds,1),1) .* exp_bound_prob, ...
                ones (size (pts(out,:),1), 1) .* exp_outside_prob);
        else
            % have a mesh, use that to compute centerline
            cent_line = horzcat (...
                (mesh(:,1)+mesh(:,3))./2, ...
                (mesh(:,2)+mesh(:,4))./2);
            % formulate interpolant
            Fr = vertcat (cent_line(4:end-4,1), mesh(:,1), mesh(:,3));
            Fc = vertcat (cent_line(4:end-4,2), mesh(:,2), mesh(:,4));
            [in, on] = inpolygon (pts(:,1), pts(:,2), ...
                [mesh(:,1);flipud(mesh(:,3))], ...
                [mesh(:,2);flipud(mesh(:,4))]);
            out = not (or (in, on));
            Fr = vertcat (Fr, pts(out,1));
            Fc = vertcat (Fc, pts(out,2));
            Fv = vertcat (zeros (size (cent_line(4:end-4,:),1), 1), ...
                ones (size (mesh,1)*2, 1) .* exp_bound_prob, ...
                ones (size (pts(out,:),1), 1) .* exp_outside_prob);
        end
        % make interpolant for this cell and apply to total image
        F = scatteredInterpolant (Fr, Fc, exp(Fv));
        for pi = 1:size(pts,1)
            cell_prob(pts(pi,1), pts(pi,2)) = max (...
                F(pts(pi,1), pts(pi,2)), ...
                cell_prob(pts(pi,1), pts(pi,2)));
        end
    end
    
    % touching mask
    % should enforce 1 pix (artificial) border b/t all cells
    [cols, rows] = meshgrid (1:img_size(2), 1:img_size(1));
    touch_mask = arrayfun (...
        @(r, c) isPixelTouchingClass (r, c, lbl_mask, 4), rows, cols);
    touch_mask = bwmorph (touch_mask, 'skel');
    cell_mask = (lbl_mask > 0) & not (touch_mask);
    
    % now generate borders for these cells
    cellbord_mask = augmentBorderClass (cell_mask, bord_strel);
    bord_mask = cellbord_mask == 2;
end