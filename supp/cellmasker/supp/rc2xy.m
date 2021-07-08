function [ varargout ] = rc2xy (img_size, varargin)
%RC2XY convert image row/col to s/y
    
    R = imref2d (img_size, [0 img_size(2)-1], [0 img_size(1)-1]);
    if nargin == 2
        A = varargin{1};
        r = A(:,1); c = A(:,2);
    else
        r = varargin{1};
        c = varargin{2};
    end
    [x, y] = intrinsicToWorld (R, c, r);
    if nargin == 2
        varargout{1} = [x, y];
    else
        varargout{1} = x;
        varargout{2} = y;
    end
end

