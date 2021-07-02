function [ xy ] = crop2globalCoords (xy, crop_box)
%CROP2GLOBALCOORDS move coordinates from cropped coordinates out to
% global (image-wide) coordinates
   if isempty(xy), return; end
   xy(:,1) = xy(:,1) + crop_box(1) - 1;
   xy(:,2) = xy(:,2) + crop_box(2) - 1;
end