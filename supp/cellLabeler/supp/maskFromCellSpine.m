function [ varargout ] = maskFromCellSpine (cell_spine, img_size, cell_width)
%MASKFROMCELLSPINE Generate a mask of a cell from the spine
%   assumes spherocylindrical shape
    orig_spine = cell_spine;
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
    [uv1, uv2] = perpVecs4Line (cell_spine(perp_filt,:));
    uv1 = uv1(1:end-1,:) .* (cell_width/2); 
    uv2 = uv2(1:end-1,:) .* (cell_width/2);
    side1 = cell_spine(not(is_cap),:) + uv1;
    side2 = cell_spine(not(is_cap),:) + uv2;
    % compute top cap
    top_vals = [side1(1,:); cell_spine(1,:); side2(1,:)]';
    angz = [0 pi/2 pi];
    cap_spln = spline (angz, top_vals);
    top_cap = ppval (cap_spln, linspace (0, pi, 20))';
    % compute bottom cap
    bot_vals = [side1(end,:); cell_spine(end,:); side2(end,:)];
    cap_spln = spline (angz, bot_vals');
    bot_cap = ppval (cap_spln, linspace (0, pi, 20))';
    % compile into shape
    rect = vertcat (top_cap, side2, flipud(bot_cap), flipud(side1));
    kill = isnan (rect(:,1)) | isnan (rect(:,2));
    rect(kill,:) = [];
    k = boundary (rect(:,1), rect(:,2), 0.8);
    mask = poly2mask (rect(:,1), rect(:,2), img_size(1), img_size(2));
    varargout{1} = mask;
    if (nargout > 1), varargout{2} = rect(k,:);
        if (nargout > 2), varargout{3} = cell_spine; end
    end
end

function [ A ] = maskArea (mask)
    A = regionprops (mask, 'Area');
    A = max (arrayfun (@(x) x.Area, A));
end