function [ Fx, Fy ] = edgeForce ( img, sigma )
%EDGEFORCE compute edge forces based on LoG algo

    if nargin < 2
        sigma = 0.01;
    end
    if isempty (sigma) || sigma < 0.01
        sigma = 0.01;
    end

    % sobel filters for x/y derivatives
    [fdx, fdy] = xyDerivFilters ();
    % make sure image is double
    if not (isa (img, 'double'))
        img = im2double (img);
    end
    img = 1 - normalizeImage (img);

    % do initial filtering
    fsize = ceil (sigma * 3) * 2 + 1;
    op = fspecial ('log', fsize, sigma);
    op = op - sum(op(:)) / numel (op);
    imf = imfilter (img, op, 'replicate');

    % extract positive side of edge
    [nr, nc] = size (img);
    rr = 2:nr-1;
    cc = 2:nc-1;
    epos = false (size (img));
    [rx, cx] = find (imf(rr,cc) > 0 & ...
                     imf(rr,cc+1) < 0 & ...
                     abs (imf(rr,cc)-imf(rr,cc+1)) > 0);
    epos((rx+1)+cx*nr) = true;
    [rx, cx] = find (imf(rr,cc) > 0 & ...
                     imf(rr,cc-1) < 0 & ...
                     abs (imf(rr,cc-1)-imf(rr,cc)) > 0);
    epos((rx+1)+cx*nr) = true;
    [rx, cx] = find (imf(rr,cc) > 0 & ...
                     imf(rr+1,cc) < 0 & ...
                     abs (imf(rr,cc)-imf(rr+1,cc)) > 0);
    epos((rx+1)+cx*nr) = true;
    [rx, cx] = find (imf(rr,cc) > 0 & ...
                     imf(rr-1,cc) < 0 & ...
                     abs (imf(rr-1,cc)-imf(rr,cc)) > 0);
    epos((rx+1)+cx*nr) = true;
    % zeros
    [rz, cz] = find (imf(rr,cc) == 0);
    if not (isempty (rz))
        isz = (rz + 1) + cz * nr;
        ind = imf(isz-1) < 0 & imf(isz+1) > 0 & ...
              abs (imf(isz-1) - imf(isz+1)) > 0;
        epos(ind) = true;
        ind = imf(isz-1) > 0 & imf(isz+1) < 0 & ...
              abs (imf(isz-1) - imf(isz+1)) > 0;
        epos(ind) = true;
        ind = imf(isz-nc) < 0 & imf(isz+nc) > 0 & ...
              abs (imf(isz-nc) - imf(isz+nc)) > 0;
        epos(ind) = true;
        ind = imf(isz-nc) > 0 & imf(isz+nc) < 0 & ...
              abs (imf(isz-nc) - imf(isz+nc)) > 0;
        epos(ind) = true;
    end
    
    % extract negative side of edge
    eneg = false (size (img));
    [rx, cx] = find (imf(rr,cc) < 0 & ...
                     imf(rr,cc+1) > 0 & ...
                     abs (imf(rr,cc)-imf(rr,cc+1)) > 0);
    eneg((rx+1)+cx*nr) = true;
    [rx, cx] = find (imf(rr,cc) < 0 & ...
                     imf(rr,cc-1) > 0 & ...
                     abs (imf(rr,cc-1)-imf(rr,cc)) > 0);
    eneg((rx+1)+cx*nr) = true;
    [rx, cx] = find (imf(rr,cc) < 0 & ...
                     imf(rr+1,cc) > 0 & ...
                     abs (imf(rr,cc)-imf(rr+1,cc)) > 0);
    eneg((rx+1)+cx*nr) = true;
    [rx, cx] = find (imf(rr,cc) < 0 & ...
                     imf(rr-1,cc) > 0 & ...
                     abs (imf(rr-1,cc)-imf(rr,cc)) > 0);
    eneg((rx+1)+cx*nr) = true;
    % zeros
    [rz, cz] = find (imf(rr,cc) == 0);
    if not (isempty (rz))
        isz = (rz + 1) + cz * nr;
        ind = imf(isz-1) > 0 & imf(isz+1) < 0 & ...
              abs (imf(isz-1) - imf(isz+1)) > 0;
        eneg(ind) = true;
        ind = imf(isz-1) < 0 & imf(isz+1) > 0 & ...
              abs (imf(isz-1) - imf(isz+1)) > 0;
        eneg(ind) = true;
        ind = imf(isz-nc) > 0 & imf(isz+nc) < 0 & ...
              abs (imf(isz-nc) - imf(isz+nc)) > 0;
        eneg(ind) = true;
        ind = imf(isz-nc) < 0 & imf(isz+nc) > 0 & ...
              abs (imf(isz-nc) - imf(isz+nc)) > 0;
        eneg(ind) = true;
    end

    % do convolution
    epos = conv2 (single (epos & not (bwmorph (epos, 'endpoints'))), ...
                  ones (3, 'single'), 'same') & epos;
    eposA = conv2 (single (epos), ones (3, 'single'), 'same') & ...
            (imf > 0) & not (epos);
    enegA = conv2 (single (eneg), ones (3, 'single'), 'same') & ...
            (imf > 0) & not (eneg);
    % compute forces
    Fx = - imf .* (imfilter (eneg+0, fdx) .* epos  - imfilter (epos+0, fdx) .* eneg + ...
                   (imfilter (epos+0, fdx) .* eposA - imfilter (enegA+0, fdx) .* enegA)./2);
    Fy = - imf .* (imfilter (eneg+0, fdy) .* epos  - imfilter (epos+0, fdy) .* eneg + ...
                   (imfilter (epos+0, fdy) .* eposA - imfilter (enegA+0, fdy) .* enegA)./2);
    % normalize forces
    Fa = vertcat (abs (Fx(:)), abs (Fy(:)));
    norm_val = mean (Fa(Fa > quantile2(Fa, 0.99))) * 2;
    Fx = Fx ./ norm_val;
    Fy = Fy ./ norm_val;
end

function [ N ] = normalizeImage ( img )
	N = double (img - min(img(:))) ./ ...
        double (max(img(:)) - min(img(:)));
end