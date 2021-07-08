function [b] = makeccw (a)
    if isempty (a)
        b = [];
    else
        if ispolycw (a(:,1), a(:,2))
            b = circshift2 (double (flipud (a)), 1);
        else
            b = a;
        end
    end
end
