function [ xyt ] = crop2cropCoords (xy, curr_box, to_box)
%CROP2CROPCOORDS map coordinates between crop boxes
%   
    xyg = crop2globalCoords (xy, curr_box);
    xyt = global2cropCoords (xyg, to_box);
end

