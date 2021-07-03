function [E_dx, E_dy, Fe_x, Fe_y] = imageForces ( im, im16, sigma_edge, ...
                                                  grad_smA, ...
                                                  thresh_min, thresh_fac, ...
                                                  grad_wgt, thr_wgt, dxy_wgt )
%IMAGEFORCES 

    % sobel filters for x/y derivatives
    [fdx, fdy] = xyDerivFilters ();

    % Edge forces
    [Fe_x, Fe_y] = edgeForce (im16, sigma_edge);
    
    % gradient forces => preliminary energy
    filt_gradsm = fspecial ('gaussian', 2 * ceil (1.5*grad_smA) + 1, grad_smA);
    img_grsm = imfilter(im, filt_gradsm);
    thresh_grad = mean (mean (img_grsm(img_grsm > mean (mean (img_grsm)))));
    energy_grad = imfilter (im, fdx).^2 + imfilter (im, fdy).^2;
    energy_grad = - energy_grad./(energy_grad + thresh_grad^2);

    % Threshold forces => preliminary energy
    lev_thr = thresh_fac * flooredGrayThresh (img_grsm, thresh_min);
    energy_thr = (img_grsm - lev_thr) .^ 2;

    % normalize by dxy (gradient)
    dx_grad = imfilter (energy_grad, fdx);
    dy_grad = imfilter (energy_grad, fdy);
    dxymax_grad = max (max (max (abs (dx_grad))), ...
                      max (max (abs (dy_grad))));
    energy_grad = energy_grad ./ dxymax_grad; 
    % normalize by dxy (threshold)
    dx_thr = imfilter (energy_thr, fdx);
    dy_thr = imfilter (energy_thr, fdy);
    dxymax_thr = max (max (max (abs (dx_thr))), ...
                      max (max (abs (dy_thr))));
    energy_thr = energy_thr ./ dxymax_thr; 

    % Combined energy
    tot_energy = (energy_grad .* grad_wgt) + (energy_thr .* thr_wgt);
    E_dx = imfilter (tot_energy, fdx) + Fe_x .* dxy_wgt;
    E_dy = imfilter (tot_energy, fdy) + Fe_y .* dxy_wgt;
end
 
