function [ outl ] = stream2outline ( x, y, width, make_cap )
% STREAMLINE2OUTLINE convert streamline to rough cell outline
    narginchk (2, 4);
    % default args
    if nargin < 4, make_cap = false;
        if nargin < 3, width = 1; end
    end
    % make sure x & y are column vectors
    A = horzcat (columnify (x), columnify (y));
    % unit vector directions of cell streamline
    dA = diff (A);
    dA = dA ./ sqrt (dA(:,1).^2 + dA(:,2).^2);

    % figure out the "straight" part of cell
    % distance from start/end of streamline
    dist_from_start = cumsum ([0; sqrt((x(2:end)-x(1:end-1)).^2 + (y(2:end)-y(1:end-1)).^2)]);
    dist_from_end  = max (dist_from_start) - dist_from_start;
    si = find (dist_from_start >= width, 1, 'first');
    ei = find (dist_from_end <= width, 1, 'first');
    
    % perpendicular vectors, scaled to width (sides of cell)
    pA1 = [dA(si:ei-1,2), -dA(si:ei-1,1)] .* width + A(si+1:ei,:);
    pA2 = [-dA(si:ei-1,2), dA(si:ei-1,2)] .* width + A(si+1:ei,:);
    if make_cap
        % now make caps
        theta1 = linspace (atan2 (-dA(si,1), dA(si,2)), ...
                           atan2 (dA(si,2), -dA(si,1)), 11);
        [c1x, c1y] = pol2cart (theta1+pi, ones(size(theta1)).*width);
        c1x = c1x + A(si+1,1);
        c1y = c1y + A(si+1,2);
        theta2 = linspace (atan2 (-dA(ei,1), dA(ei,2)), ...
                           atan2 (dA(ei,2), -dA(ei,1)), 11);
        [c2x, c2y] = pol2cart (theta2, ones(size(theta2)).*width);
        c2x = c2x + A(ei,1);
        c2y = c2y + A(ei,2);
    else
        c1x = []; c1y = [];
        c2x = []; c2y = [];
    end
    outl = horzcat (...
        vertcat (c1x', pA1(:,1), c2x', pA2(end:-1:1,1)), ...
        vertcat (c1y', pA1(:,2), c2y', pA2(end:-1:1,2)));
    outl = vertcat (outl, outl(1,:));  % so that it connects at the end
end

function [ cvec ] = columnify ( vec )
    if size (vec, 2) > size (vec, 1)
        cvec = transpose (vec);
    else
        cvec = vec;
    end
end