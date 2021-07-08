function [ varargout ] = xy2rc (img_size, varargin)
%RC2XY convert image x/y coordinates to row/col
    R = imref2d (img_size, [0 img_size(2)-1], [0 img_size(1)-1]);
    
    if nargin == 2
        A = varargin{1};
        x = A(:,1); y = A(:,2);
    else
        x = varargin{1};
        y = varargin{2};
    end
    [c, r] = worldToIntrinsic (R, x, y);
    if nargin == 2
        varargout{1} = [r, c];
    else
        varargout{1} = r;
        varargout{2} = c;
    end
end

