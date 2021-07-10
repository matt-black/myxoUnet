function [ varargout ] = maskFromCellSpine (cell_spine, img_size, cell_width)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

    dcell_spine = diff (cell_spine(1:end-1,:));
    uv = arrayfun (@(r) dcell_spine(r,:) ./ sqrt (sum (dcell_spine(r,:).^2)), ...
        (1:size(dcell_spine,1))', 'UniformOutput', false);
    uv = cat (1, uv{:});
    ang = arrayfun (@(r) atan2 (uv(r,2), uv(r,1)), (1:size(uv,1))');
    uv1 = arrayfun (@(a) horzcat (cos (a+pi/2), sin (a+pi/2)), ang, ...
        'UniformOutput', false);
    uv1 = cat (1, uv1{:}) .* (cell_width / 2);
    uv2 = arrayfun (@(a) horzcat (cos (a-pi/2), sin (a-pi/2)), ang, ...
        'UniformOutput', false);
    uv2 = cat (1, uv2{:}) .* (cell_width / 2) ;

    rect = vertcat (cell_spine(1,:), cell_spine(2:end-1,:)+uv1, ...
        cell_spine(end,:), flipud (cell_spine(2:end-1,:)+uv2));

    mask = poly2mask (rect(:,1), rect(:,2), img_size(1), img_size(2));
    
    varargout{1} = mask;
    if nargout > 1
        varargout{2} = rect;
    end
end

