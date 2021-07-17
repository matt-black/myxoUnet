function [ resampled ] = resampleCellSpine (spine, d_samp)
%RESAMPLECELLSPINE resample spine points with constant distance-spacing
    spn_cell = num2cell (spine, 2);
    % compute arc length at each point on original spine
    ptdists = cellfun (@(p1,p2) sqrt(sum((p1-p2).^2)), ...
        spn_cell(1:end-1), spn_cell(2:end));
    arclen = [0; cumsum(ptdists)];
    kill = [];
    for ii = 2:length(arclen)
        if arclen(ii) == arclen(ii-1)
            kill = vertcat (kill, ii);
        end
    end
    arclen(kill) = [];
    spine(kill,:) = [];
    len_d = linspace (0, arclen(end), length (0:d_samp:arclen(end)));
%     x = interp1 (arclen, spine(:,1), len_d);
%     y = interp1 (arclen, spine(:,2), len_d);
%     resampled = [x; y]';
    % fit to spline for upsampling/interpolation
    spln = spline (arclen, spine');
    resampled = transpose (ppval (spln, len_d));  % do resampling
end