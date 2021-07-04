function [ mask, cells ] = cells2mask ( cells, img_size, label, expand )
%CELLS2MASK convert list of cells to a mask for image of size `img_size`
    mask = zeros (img_size);
    
    %% conversion
    % convert x/y of cells to r/c of pixel mask
    % object for converting x/y to r/c
    rconv = imref2d (img_size, [0 img_size(2)-1], [0 img_size(1)-1]);
    % loop through cells, convert each
    for ic = 1:numel(cells)
        [row, col] = worldToSubscript (rconv, cells(ic).pix(:,1), ...
                                       cells(ic).pix(:,2));
        % convert r/c to linear index to assign
        ind = sub2ind (img_size, row, col);
        ind(ind<1|isnan(ind)) = [];  % only want valid assignments
        mask(ind) = ic;
        cells(ic).id = ic;
    end
    
    %% expansion
    % each loop tries to expand the mask of each cell by 1 pixel
    for ei = 1:expand
        for ii = 1:max(mask(:))  % loop over labels
            [sr, sc] = find (mask == ii);
            cands = arrayfun (@(r,c) getNhood(r,c,size(mask)), sr, sc, ...
                'UniformOutput', false);
            cands = unique (vertcat (cands{:}));
            cands(mask(cands)>0) = [];
            mask(cands) = ii;
        end
    end
    if not (label)
        mask = mask > 0;
    end
end

function [ varargout ] = getNhood ( row, col, img_sze )
    A = [row-1 , col-1;
         row-1 , col;
         row-1 , col+1;
         row   , col-1;
         row   , col+1;
         row+1 , col-1;
         row+1 , col;
         row+1 , col+1];
    % kill negative rows/cols
    neg_rc = A(:,1) < 1 | A(:,2) < 1;
    pos_rc = A(:,1) > img_sze(1) | ...
             A(:,2) > img_sze(2);
    A(neg_rc|pos_rc,:) = [];
    if nargout == 2
        varargout{1} = A(:,1);
        varargout{2} = A(:,2);
    elseif nargout == 1
        varargout{1} = sub2ind (img_sze, A(:,1), A(:,2));
    else
        error ('getNhood :: too many varargout');
    end 
end