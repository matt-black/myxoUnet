function [ dX, dY ] = xyDerivFilters ( )
% XYDERIVFILTERS generates sobel filters for taking x/y derivatives
    dX = fliplr (fspecial ('sobel')');
    dY = fspecial ('sobel');
end