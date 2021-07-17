function [ uv1, uv2 ] = perpVecs4Line ( xy, skip, isrowcol )
    if nargin < 3, isrowcol = false;
        if nargin < 2, skip = 1;
        end
    end
    if (isrowcol)                       % flip to xy
        xy = [xy(:,2), xy(:,1)];
    end
    if skip == 1, dxy = diff (xy);
    else
        len = size (xy, 1);
        dxy = arrayfun (@(ii) xy(ii,:) - xy(ii+skip,:), 1:(len-skip), ...
                        'UniformOutput', false)';
        dxy = cell2mat (dxy);        
    end
    
    % normalize to unit vector
    for r = 1:size(dxy,1)
        dxy(r,:) = dxy(r,:) ./ abs (complex (dxy(r,1), dxy(r,2)));
    end
    ang = arrayfun (@(r) atan2 (dxy(r,2), dxy(r,1)), (1:size(dxy,1))');
    if length (ang) > 1
        % interpolate to get angle at last point
        F = scatteredInterpolant (xy(1:end-1,1), xy(1:end-1,2), ang);
        ang = vertcat (F(xy(end,:)), ang);
        % rotate +/- 90 degrees
        ang1 = ang + pi/2; ang2 = ang - pi/2;
    else
        ang1 = [ang;ang] + pi/2;
        ang2 = [ang;ang] - pi/2;
    end
    uv1 = [cos(ang1), sin(ang1)];
    uv2 = [cos(ang2), sin(ang2)];
end