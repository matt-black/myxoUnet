function [ varargout ] = maskFromCellSpine2 (cell_spine, img, cell_width)
%MASKFROMCELLSPINE Generate a mask of a cell from the spine
%   using brightest pixels within cell_width of spine.
%%
    img_size = size(img);
    % resample spine
    resamp_dist = cell_width / 6;
    cell_spine = resampleCellSpine (cell_spine, resamp_dist);
    % figure out which are cap/body regions
%     d_from_start = cellfun (@(p) sqrt(sum((p-cell_spine(1,:)).^2)), ...
%         num2cell (cell_spine, 2));
%     d_from_end = cellfun (@(p) sqrt(sum((p-cell_spine(end,:)).^2)), ...
%         num2cell (cell_spine, 2));
%     is_cap = d_from_start <= cell_width/2 | d_from_end <= cell_width/2;
    % compute perpendicular vectors for body region
    % NOTE: take extra point at end so that we don't have to interpolate
    % vector at the end of the body
%     perp_filt = not (is_cap);
%     if sum(perp_filt) == 0
%         perp_filt(floor(end/2)) = true;
%     end
%     perp_filt(find(perp_filt,1,'last')+1) = true;
    perp_n = round(cell_width/0.25);
    [uv1, ~] = perpVecs4Line (cell_spine);
    % Find x and y values along lines perpendicular to each point on cell
    % spine.
    uv_caps = [cos([0:pi/8:pi]'), sin([0:pi/8:pi]')];
    uv1 = [uv_caps; uv1; uv_caps];
    pt_is = -cell_width:0.25:cell_width;
    pt_xs = pt_is'*uv1(:,1)';
    pt_ys = pt_is'*uv1(:,2)';
    perp_line_centers_xs = [repelem(cell_spine(1,1),numel(pt_is),numel(uv_caps(:,1))), ...
        repelem(cell_spine(:,1)',numel(pt_is),1) ...
        repelem(cell_spine(end,1),numel(pt_is),numel(uv_caps(:,1)))];
    perp_line_xs = pt_xs + perp_line_centers_xs;
    perp_line_xs(perp_line_xs<1) = 1;
    perp_line_xs(perp_line_xs>img_size(2)) = img_size(2);
    
    perp_line_centers_ys = [repelem(cell_spine(1,2),numel(pt_is),numel(uv_caps(:,2))), ...
        repelem(cell_spine(:,2)',numel(pt_is),1) ...
        repelem(cell_spine(end,2),numel(pt_is),numel(uv_caps(:,2)))];
    perp_line_ys = pt_ys + perp_line_centers_ys;
    perp_line_ys(perp_line_ys<1) = 1;
    perp_line_ys(perp_line_ys>img_size(2)) = img_size(2);
    
    % Take values of original image at each point on all perpendicular lines.
    
    inds = sub2ind(img_size,round(perp_line_ys),round(perp_line_xs));
    line_ls = img(inds);
    
    % Find the maximum gaussian weighted average image values for each perp line. 
    ls_filt = fspecial('gaussian',[perp_n, 1],perp_n/2);
    mean_ls = conv2(line_ls,ls_filt,'same');
    mean_ls(1:ceil(perp_n/2),:) = 0;
    mean_ls(end-ceil(perp_n/2)+1:end,:) = 0;
    [~,max_ind] = max(mean_ls);
    
    % Map maximum brightness back to pixels on the original image.
    ind_mask = zeros(size(inds));
    ind_mask(sub2ind(size(inds),max_ind,1:numel(max_ind))) = 1;
    body_inds = conv2(ind_mask,ones(perp_n,1),'same')==1;
    cell_xs = perp_line_xs(body_inds);
    cell_ys = perp_line_ys(body_inds);
    k = boundary(cell_xs,cell_ys);
    
    mask = poly2mask(cell_xs(k),cell_ys(k),img_size(1),img_size(2));
    
    % Fill in the mask for the cell body.
    
    varargout{1} = mask;
    
