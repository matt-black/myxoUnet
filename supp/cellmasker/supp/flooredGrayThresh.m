function [thr_val] = flooredGrayThresh (img, floor_lvl)
%GRAYTHRESHOLD image threshold value but only for image values above floor
    if nargin < 2, floor_lvl = 0; end
    % subset out edges
    [nr,nc] = size(img);
    row_subs = (ceil (nr/20)):(floor (nr*0.95));
    col_subs = (ceil (nc/20)):(floor (nc*0.95));
    % get threshold value
    thr_val = doThresh (img(row_subs,col_subs), floor_lvl);
end

function [thr] = doThresh (a, floor_lvl)
    if floor_lvl>0
        c = reshape(a,1,[]);
        c = sort(c);
        level = c(ceil(min(floor_lvl,1)*length(c)));
        thr = graythresh(c(c>=level));
    else
        thr = graythresh(a);
    end
end