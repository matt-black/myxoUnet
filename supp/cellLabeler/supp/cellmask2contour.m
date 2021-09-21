function [ contour ] = cellmask2contour ( mask, fsmooth, mesh_step, mesh_tol, mesh_width )
%CELLMASK2CONTOUR generate (rough) cell contour from mask of single cell
    narginchk (2, 5);
    % trace out boundary of cell
    [ri,ci] = find (bwperim (mask), 1, 'first');
    boundry = bwtraceboundary (mask, [ri,ci], 'n', 4, Inf, 'counterclockwise');
    % smooth boundary in fourier space
    contour_prelim = smoothContour (boundry, fsmooth);
    if nargin > 2
        % make mesh
        mesh = model2MeshForRefine (contour_prelim, mesh_step, mesh_tol, mesh_width);
        if length (mesh) > 4
            contour = fliplr ([mesh(:,1:2); flipud(mesh(:,3:4))]);
        else
            contour = [];
        end
        contour = makeccw (contour);
    else
        contour = makeccw (contour_prelim);
    end
end

function b = makeccw(a)
    if isempty(a)
        b = [];
    else
        if ispolycw (a(:,1), a(:,2))
            b = circshift2(double(flipud(a)),1);
        else
            b=a;
        end
    end
end

function c=circshift2(a,b)
    L=size(a,1);
    c = a(mod((0:L-1)-b,L)+1,:);
end

