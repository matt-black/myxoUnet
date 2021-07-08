function [ img ] = normalizeImage (img_in)
%NORMALIZEIMAGE normalize image values to range [0,1]
    if not (isa (img_in, 'double'))
        % convert to double
        img_in = im2double (img_in);
    end
    min_val = min (img_in(:));
    max_val = max (img_in(:));
    img = (img_in - min_val) ./ ...
        (max_val - min_val);
end

