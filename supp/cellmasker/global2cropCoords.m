function [ xy ] = global2cropCoords (xy, crop_box)
%GLOBAL2CROPCOORDS convert global coordinates to cropped region
   if isempty(xy), return; end
   xy(:,1) = xy(:,1) - crop_box(1) + 1;
   xy(:,2) = xy(:,2) - crop_box(2) + 1;
end

