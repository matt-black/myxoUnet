function [ varargout ] = maskFromCellSpine2 (cell_spine, img, cell_width)
%MASKFROMCELLSPINE Generate a mask of a cell from the spine
%   using brightest pixels within cell_width of spine.
%%
    img_size = size(img);
    % resample spine
    resamp_dist = cell_width / 6;
    cell_spine = resampleCellSpine (cell_spine, resamp_dist);
    % figure out which are cap/body regions
    d_from_start = cellfun (@(p) sqrt(sum((p-cell_spine(1,:)).^2)), ...
        num2cell (cell_spine, 2));
    d_from_end = cellfun (@(p) sqrt(sum((p-cell_spine(end,:)).^2)), ...
        num2cell (cell_spine, 2));
    is_cap = d_from_start <= cell_width/2 | d_from_end <= cell_width/2;
    % compute perpendicular vectors for body region
    % NOTE: take extra point at end so that we don't have to interpolate
    % vector at the end of the body
    perp_filt = not (is_cap);
    perp_filt(find(perp_filt,1,'last')+1) = true;
    perp_n = round(cell_width/0.25 - 3);
    [uv1, ~] = perpVecs4Line (cell_spine(perp_filt,:));
    % Find x and y values along lines perpendicular to each point on cell
    % spine.
    pt_is = -cell_width:0.25:cell_width;
    pt_xs = pt_is'*uv1(:,1)';
    pt_ys = pt_is'*uv1(:,2)';
    perp_line_xs = pt_xs + repelem(cell_spine(perp_filt,1)',numel(pt_is),1);
    perp_line_xs(perp_line_xs<1) = 1;
    perp_line_xs(perp_line_xs>img_size(2)) = img_size(2);
    perp_line_ys = pt_ys + repelem(cell_spine(perp_filt,2)',numel(pt_is),1);
    perp_line_ys(perp_line_ys<1) = 1;
    perp_line_ys(perp_line_ys>img_size(1)) = img_size(1);
    
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
    
    % Fill in the mask for the cell body (without caps yet).
    mask = zeros(img_size);
    mask(inds(body_inds))=1;
    
    % Find cell caps.
    % Top cap.
    top_ind_1 = find(body_inds(:,1),1,'first');
    top_ind_2 = find(body_inds(:,1),1,'last');
    
    top_vals = [perp_line_xs(top_ind_1,1) perp_line_ys(top_ind_1,1); ...
         cell_spine(1,:); ...
         perp_line_xs(top_ind_2,1) perp_line_ys(top_ind_2,1)]';
     
    angz = [0 pi/2 pi];
    cap_spln = spline (angz, top_vals);
    top_cap = ppval (cap_spln, linspace (0, pi, 20))';
    top_mask = poly2mask(top_cap(:,1),top_cap(:,2),img_size(1),img_size(2));
    
    % Bottom cap.
    bot_ind_1 = find(body_inds(:,end),1,'first');
    bot_ind_2 = find(body_inds(:,end),1,'last');
    
    bot_vals = [perp_line_xs(bot_ind_1,end) perp_line_ys(bot_ind_1,end); ...
         cell_spine(end,:); ...
         perp_line_xs(bot_ind_2,end) perp_line_ys(bot_ind_2,end)]';
     
    angz = [0 pi/2 pi];
    
    cap_spln = spline (angz, bot_vals);
    bot_cap = ppval (cap_spln, linspace (0, pi, 20))';
    bot_mask = poly2mask(bot_cap(:,1),bot_cap(:,2),img_size(1),img_size(2));

    mask = round(imgaussfilt(mask,1));
    mask(top_mask) = 1;
    mask(bot_mask) = 1;
    
    varargout{1} = mask;
    
