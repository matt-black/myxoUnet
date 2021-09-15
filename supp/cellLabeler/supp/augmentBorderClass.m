function [ varargout ] = augmentBorderClass( cell_mask, se )
%MAKEBORDERCLASS 
    if nargin == 1, se = strel ('square', 3); end
    img_sze = size (cell_mask);
    % trace boundary of cells & convert to pixelIdx form
    bnd = bwboundaries (cell_mask, 4, 'noholes');
    bnd = cellfun (@(A) sub2ind (img_sze, A(:,1), A(:,2)), bnd, ...
        'UniformOutput', false);
    % formulate dummy connected components object
    cc = struct ('Connectivity', 4, ...
        'ImageSize', img_sze, ...
        'NumObjects', length(bnd));
    cc.PixelIdxList = bnd;
    % dilate boundary
    bord_mask = uint8 (imdilate (labelmatrix (cc) > 0, se));
    % set border values to 2, then let cell class (=1) take priority
    % so that cell interiors are marked as cell, only exterior values
    % marked as border
    bord_mask(bord_mask>0) = 2;
    bord_mask(cell_mask==1) = 1;
    varargout{1} = bord_mask;
    if nargout > 1
        % generate distance image, if specified
        d2b = bwdist (bord_mask == 2, 'euclidean');
        d2b(~cell_mask) = 0;
        varargout{2} = d2b;
    end
end

