function [contour, fitqual] = refineContour (im, E_dx, E_dy, thres, contour, cell_width, max_area, ...
                                             rigid_range, rigid_rangeB, rigidity, rigidityB, springconst, ...
                                             scale_factor, fitstep, fitmaxiter, ...
                                             img_force, attr_pwr, attr_coeff, rep_coeff, ...
                                             neigh_rep, thres_facF, ...
                                             horz_align, eq_dist, move_all)
    %REFINECONTOUR

    % rigid_range : 2.5
    % rigid_rangeB : 7-8
    % horz_align : 0.2
    % eq_dist : 2.5
    fitqual = 0;
    if isempty(contour) || length(contour(:,1)) < 10 || isempty(im)
        contour = []; fitqual = 0;
        return;
    end
    % ensure odd mesh length
    if mod(length(contour(:,1)),2)
        contour = contour(1:end-1,:);
    end 
    contour = double (makeccw (contour));

    % Initializations
    L = size (contour, 1); N = ceil (L/2) + 1;
    stp = 1;
    H = ceil (cell_width * pi / 2 / stp / 2);
    xCell = contour(:,1); yCell = contour(:,2); Kstp = 1;
    
    % Construct model cell and the rigidity forces in it
    ddx = round(rigid_range);
    A = 1/2./(1:ddx); A = A/(2*sum(A)-A(1));
    ddxB = round(rigid_rangeB);
    B = 1/2./sqrt(1:ddxB); B = B/(2*sum(B)-B(1));
    HA = H+1:-1:1; HA = pi*HA/(2*sum(HA)-HA(1)); % ???
    x(H+1+2*ddx) = 0; y(H+1+2*ddx) = 0;   
    for i=H+2*ddx:-1:H+1
        x(i) = x(i+1) - stp; y(i) = y(i+1);  
    end     
    alpha = HA(H+1);
    for i=H:-1:1
        x(i) = x(i+1) - stp*cos(alpha);
        y(i) = y(i+1) + stp*sin(alpha);
        alpha = HA(i) + alpha;
    end
    x = [fliplr(x(2:end)),x];
    y = [2*y(1)-fliplr(y(2:end)),y];
    y = y*cell_width*scale_factor/abs(y(1));
    [fx,fy] = getRigidForces (x',y',A);
    [Tx,Ty] = getNormals (x',y');
    f = Tx.*fx + Ty.*fy;
    f = f((end+1)/2:(end+1)/2+H);
    hcorr = [f;zeros(N-2*H-2,1);flipud(f);f(2:end);zeros(N-2*H-2,1);flipud(f(2:end))];
    rgt = [1:H+1 (H+1)*ones(1,N-2*H-2) H+1:-1:1 2:H+1 (H+1)*ones(1,N-2*H-2) H+1:-1:2]'/(H+1);
    if length(rgt)>L % N-2
        L2 = length(rgt); 
        rm = ceil((L2-L)/2);
        lv = ceil(N/2)-1;
        hcorr = hcorr([1:lv lv+1+rm:L2/2+1+lv L2/2+1+lv+1+rm:end]);
        rgt = rgt([1:lv lv+1+rm:L2/2+1+lv L2/2+1+lv+1+rm:end]); 
    end
    % Opposite points interaction (Fdstx,Fdsty)
    lside = 2:N-1;
    rside = L:-1:N+1;
    
    cellWidth = [2*abs(y(end/2+3/2:end/2+1/2+H) - y(end/2+1/2))';
                 cell_width*scale_factor*ones(N-2*H-2,1);
                 2*abs(y(end/2+1/2+H:-1:end/2+3/2)-y(end/2+1/2))'];
    if length (cellWidth) > N - 2
        cellWidth = cellWidth([1:ceil(N/2)-1 end+2-floor(N/2):end]);
    end
    wdt = cellWidth/max(cellWidth);
    
    fitqualHistory = 0;
    areaCell = [];
    fitqualHistoryCounter = 0;
    
    if img_force >= 7
        fitqualThresh = 13;
    else
        fitqualThresh = 21;
    end

    for a=1:fitmaxiter
        % Vector image forces (Fix,Fiy)
        Fix = -img_force * interp2 (E_dx,double(xCell),double(yCell),'linear',0);
        Fiy =  img_force * interp2 (E_dy,double(xCell),double(yCell),'linear',0);
        % Get normals to the centerline (Tx,Ty)
        Tx = circshift2(yCell,-1) - circshift2(yCell,1);
        Ty = circshift2(xCell,1) - circshift2(xCell,-1);
        dtxy = sqrt(Tx.^2 + Ty.^2);
        dtxy = dtxy + mean (dtxy, 'omitnan')/100;
        if min(dtxy)==0, contour=[];fitqual=0;return;end
        Tx = Tx./dtxy;
        Ty = Ty./dtxy;
        
        Lx = Ty;  Ly = -Tx;
        % Get area outside the cell & attraction/repulsion (Fax,Fay)
        % Matrix attRegion wide outside
        TxM = repmat (xCell,1,2*1+1)+Tx*(-1:1);
        TyM = repmat (yCell,1,2*1+1)+Ty*(-1:1);
        % Matrix attRegion wide outside
        Clr0 = interp2 (im,TxM,TyM,'linear',0);
        
        % Non-normalized 'area' attraction
        Clr = 1-1./(1+(Clr0/thres/thres_facF).^attr_pwr);
        
        % Cell area
        are = polyarea (xCell,yCell);
        % Scalar repulsion/attraction forces
        T = attr_coeff * mean (Clr(:,1+1:end), 2, 'omitnan') ...
            - rep_coeff * (are<0.9*max_area) * (1 - mean (Clr(:,1:1+1), 2, 'omitnan'));
        
        % Vector repulsion/attraction forces
        Fax = Tx.*T;
        Fay = Ty.*T;
        T = - neigh_rep * interp2 (E_dx*0,xCell,yCell,'linear',0);
        Fnrx = Tx.*T;  Fnry = Ty.*T;
        
        % Opposite points interaction (Fdstx,Fdsty)
        Dst = sqrt((xCell(lside)-xCell(rside)).^2 + (yCell(lside)-yCell(rside)).^2);
        Fdst = (cellWidth-Dst)./cellWidth;
        g = 5;
        Fdst((Dst./cellWidth)<0.5)=Fdst((Dst./cellWidth)<0.5).*g-(g-1)*0.5;
        Fdst = springconst*wdt.*Fdst.*cellWidth;
        Fdst1x = Fdst.*(xCell(lside)-xCell(rside))./Dst;
        Fdst1y = Fdst.*(yCell(lside)-yCell(rside))./Dst;
        Fdstx = zeros(L,1); Fdsty = zeros(L,1);
        Fdstx(lside) = Fdst1x; Fdsty(lside) = Fdst1y;
        Fdstx(rside) = -Fdst1x; Fdsty(rside) = -Fdst1y;
        
        % Rigidity (Frx,Fry)
        [D4x,D4y] = getRigidForces (double(xCell), double(yCell), A);
        Frx = rigidity * (D4x - Tx.*hcorr).*rgt;
        Fry = rigidity * (D4y - Ty.*hcorr).*rgt;
        
        % Backbone rigidity (Fbrx,Fbry)
        xCnt = (xCell(1:N)+flipud(xCell([N:L 1])))/2;
        yCnt = (yCell(1:N)+flipud(yCell([N:L 1])))/2;
        Tbx = circshift2(yCnt,-1) - circshift2(yCnt,1);
        Tby = circshift2(xCnt,1) - circshift2(xCnt,-1);
        Tbx(end) = Tbx(end-1); Tby(end) = Tby(end-1);
        Tbx(1) = Tbx(2); Tby(1) = Tby(2);
        dtxy = sqrt(Tbx.^2 + Tby.^2);
        Tbx = Tbx./dtxy; Tby = Tby./dtxy;
        
        [D4btx,D4bty] = getRigidForcesL(double(xCnt),double(yCnt),B);
        
        D4b = (D4btx.*Tbx + D4bty.*Tby)/2;
        D4bx = rigidityB * D4b.*Tbx; D4by = rigidityB * D4b.*Tby;
        Fbrx = [D4bx;flipud(D4bx(2:end-1))]; Fbry = [D4by;flipud(D4by(2:end-1))];
        
        % Perpendicular ribs (Fpx,Fpy)
        Fpbx = xCell(lside)-xCell(rside); Fpby = yCell(lside)-yCell(rside);
        Fp = Fpbx.*Tbx(2:end-1) + Fpby.*Tby(2:end-1);
        Fpx = zeros(L,1); Fpy = zeros(L,1);
        Fpbx = horz_align*(Fpbx-Fp.*Tbx(2:end-1));
        Fpby = horz_align*(Fpby-Fp.*Tby(2:end-1));
        Fpx(lside) = -Fpbx; Fpy(lside) = -Fpby;
        Fpx(rside) = Fpbx; Fpy(rside) = Fpby;
        
        % Equal distances between points (Fqx,Fqy)
        Fqx = eq_dist*(circshift2(xCell,1)+ circshift2(xCell,-1)-2*xCell);
        Fqy = eq_dist*(circshift2(yCell,1)+ circshift2(yCell,-1)-2*yCell);
        Fq = Lx.*Fqx + Ly.*Fqy;
        Fqx = Fq.*Lx; Fqy = Fq.*Ly;
        
        % Get the resulting force
        if a>1, Fo = [Fx;Fy]; end
        if isempty(who('Kstp2'))||isempty(Kstp2), Kstp2 = 1; end
        Fx = (Fix + Fax + Fnrx + Fdstx) + Frx;
        Fy = (Fiy + Fay + Fnry + Fdsty) + Fry;
        Fs = Fx.*Tx + Fy.*Ty;
        Fx = Fs.*Tx + Fpx + Fbrx + Fqx;
        Fy = Fs.*Ty + Fpy + Fbry + Fqy;
        % Normalize
        Fm = abs(Fs).^0.2;
        Fm = Fm + mean (Fm, 'omitnan')/100;
        
        if min(Fm)==0, disp('Error - zero force'); contour=[]; fitqual = 0; return; end
        if a>1
            K = sum((Fo.*[Fx;Fy])>0)/2/L;
            if K<0.4
                Kstp=Kstp/1.4; % Kstp saturates oscillations perpendicular to the coutour
            elseif K>0.6
                Kstp=min(1,Kstp.*1.2);
            end
        end
        mxf = fitstep*Kstp*max(max(abs(Fx))+max(abs(Fy)));
        asd = (xCell - circshift2(xCell,1)).^2+(yCell- circshift2(yCell,1)).^2;
        mnd = sqrt (min (asd));
        med = sqrt (mean (asd, 'omitnan'));
        
        % Kstp2 prevents points crossing
        Kstp2 = min([Kstp2*1.1 1 mnd/mxf/2 3*mnd/med]);
        if move_all>0
            if a>1
                mfxold = mfx;
                mfyold = mfy;
                MFold = MF;
            end
            xm = xCell - mean(xCell, 'omitnan'); ym = yCell - mean(yCell, 'omitnan');
            MF = mean (-Fy.*xm+Fx.*ym, 'omitnan'); MI = sum (xm.^2+ym.^2);
            Fmrx =  ym*MF/MI; Fmry = -xm*MF/MI;
            mfx = mean (Fx, 'omitnan'); mfy = mean (Fy, 'omitnan');

            mfitstep = fitstep;
            Kstpm = mfitstep;
            % Kstpm prevents large (>1px) mean steps
            Kstpm = min([Kstpm*1.5 mfitstep/(sqrt(mean(Fx, 'omitnan')^2 + mean(Fy, 'omitnan')^2)), ...
                         mfitstep/abs(MF)*sqrt(MI)]);
            if a>1 && (mfx*sign(mfxold)<-abs(mfxold)/2 || mfy*sign(mfyold)<-abs(mfyold)/2 || MF*sign(MFold)<-abs(MFold)/2)
                Kstpm = Kstpm/2;
            end
        end
        
        % Move
        if move_all>0
            Fx = Kstp*Kstp2*scale_factor*fitstep*Fx*(1-move_all)+Kstpm*(mean(Fx, 'omitnan')+Fmrx)*move_all;
            Fy = Kstp*Kstp2*scale_factor*fitstep*Fy*(1-move_all)+Kstpm*(mean(Fy, 'omitnan')+Fmry)*move_all;
        else
            Fx = Kstp*Kstp2*scale_factor*fitstep*Fx;
            Fy = Kstp*Kstp2*scale_factor*fitstep*Fy;
        end
        xCell = xCell + Fx; yCell = yCell + Fy;
        if max(isnan(xCell))==1, contour=[]; fitqual=0; return; end
        
        % Looking for self-intersections
        [i1, i2] = selfIntersect (double(xCell), double(yCell));
        % Moving points halfway to the projection on the opposite strand
        iMovCurveArr = []; xMovCurveArr = []; yMovCurveArr = [];

        for i=1:2:(length(i1)-1)
            if i1(i)<=i1(i+1)
                iMovCurve = mod((i1(i)+1:i1(i+1))-1,L)+1;
            else
                iMovCurve = mod((i1(i)+1:i1(i+1)+L)-1,L)+1;
            end
            if length(iMovCurve)<2, continue; end
            if i2(i)+1>=i2(i+1)
                iRefCurve = mod((i2(i)+1:-1:i2(i+1))-1,L)+1;
            else
                iRefCurve = mod((i2(i)+1+L:-1:i2(i+1))-1,L)+1;
            end
            % iMovCurve = mod((i1(i)+1:i1(i+1))-1,L)+1;
            % if length(iMovCurve)<2, continue; end
            % iRefCurve = mod((i2(i)+1:-1:i2(i+1))-1,L)+1;
            xMovCurve = reshape(xCell(iMovCurve),1,[]);
            yMovCurve = reshape(yCell(iMovCurve),1,[]);
            xRefCurve = reshape(xCell(iRefCurve),1,[]);
            yRefCurve = reshape(yCell(iRefCurve),1,[]);
            [xMovCurve,yMovCurve]=projectCurve(xMovCurve,yMovCurve,xRefCurve,yRefCurve);
            iMovCurveArr = [iMovCurveArr iMovCurve];
            xMovCurveArr = [xMovCurveArr xMovCurve];
            yMovCurveArr = [yMovCurveArr yMovCurve];
        end

        xCell(iMovCurveArr) = xMovCurveArr;
        yCell(iMovCurveArr) = yMovCurveArr;
        
        % Condition to finish
        fitqual = sum (abs (Fm)) / length (Fm) /2;
        %------------------------------------------------------------------------
        %check for amount of change in fitqual value compare to previous 10
        %frames if the value is greater than change of 0.01 then keep
        %looping otherwise break the loop to save processing time and
        %randomness of image force minimizing function.  Ahmad.P Sept 18.
        %2012.
        fitQualitySumOfForces = sum (abs (Fs)) / (length (Fs) / 2);
        if fitQualitySumOfForces < 0, break; end
    end 

    % Output model
    contour = [xCell,yCell];
    % if isfield(p,'smoothafterfit') && p.smoothafterfit
    %     fpp = frdescp(contour);
    %     cCell = ifdescp(fpp,p.fsmooth);
    %     mesh = model2MeshForRefine(double(cCell),p.fmeshstep,p.meshTolerance,p.meshWidth);
    %     if length(mesh)>4
    %         contour = [mesh(:,1:2);flipud(mesh(2:end-1,3:4))];
    %     else
    %         contour = [];
    %     end
    %     contour = makeccw(contour);
    % end
end

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

function [c] = circshift2 (a, b)
    L = size (a, 1);
    c = a(mod ((0:L-1)-b,L)+1,:);
end

function [F_x, F_y] = getRigidForcesL (img_x, img_y, wgt_vec)
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%function [F_x,F_y] = getRigidForcesL(img_x,img_y,wgt_vec)
%oufti v0.3.1
%@author:  Oleksii Sliusarenko
%@modified:    July 19 2013(ap)
%@copyright 2012-2013 Yale University
%==========================================================================
%**********output********
%F_x:    forces in the x-direction
%F_y:    forces inthe y-direction
%**********Input********
%img_x:        image values in the x-axis
%img_y:        image values in the y-axis
%wgt_vec:  weight values to calculate image forces.
%**********Purpose******
%The function computes image forces from given image values using provided
%weight vector.
%==========================================================================
    F_x = zeros(size(img_x,1),1);
    F_y = zeros(size(img_y,1),1);
    for i = 1:length(wgt_vec)
        fxt = wgt_vec(i)*(img_x(1:end-2*i)/2+img_x(2*i+1:end)/2-img_x(i+1:end-i));
        fyt = wgt_vec(i)*(img_y(1:end-2*i)/2+img_y(2*i+1:end)/2-img_y(i+1:end-i));
        F_x(i+1:end-i) = F_x(i+1:end-i) + fxt;
        F_y(i+1:end-i) = F_y(i+1:end-i) + fyt;
        F_x(1:end-2*i) = F_x(1:end-2*i) - fxt/2;
        F_y(1:end-2*i) = F_y(1:end-2*i) - fyt/2;
        F_x(2*i+1:end) = F_x(2*i+1:end) - fxt/2;
        F_y(2*i+1:end) = F_y(2*i+1:end) - fyt/2;
    end
end

function [F_x,F_y] = getRigidForces(img_x,img_y,wgt_vec)
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%function [F_x,F_y] = getRigidForcesL(img_x,img_y,wgt_vec)
%oufti v0.3.1
%@author:  Oleksii Sliusarenko
%@modified:    July 19 2013(ap)
%@copyright 2012-2013 Yale University
%==========================================================================
%**********output********
%F_x:    forces in the x-direction
%F_y:    forces inthe y-direction
%**********Input********
%img_x:        image values in the x-axis
%img_y:        image values in the y-axis
%wgt_vec:  weight values to calculate image forces.
%**********Purpose******
%The function computes image forces from given image values using provided
%weight vector.
%==========================================================================
    F_x = - 2*sum(wgt_vec)*img_x;
    F_y = - 2*sum(wgt_vec)*img_y;
    for i=1:length(wgt_vec)
        F_x = F_x + wgt_vec(i)*(circshift2(img_x,i)+circshift2(img_x,-i));
        F_y = F_y + wgt_vec(i)*(circshift2(img_y,i)+circshift2(img_y,-i));
    end
end

function [tx, ty] = getNormals (x, y)
    tx = circshift2(y,-1) - circshift2(y,1);
    ty = circshift2(x,1) - circshift2(x,-1);
    dtxy = sqrt(tx.^2 + ty.^2);
    tx = tx./dtxy;
    ty = ty./dtxy;
end


function [ i_out, j_out ] = selfIntersect (x,y)
%SELFINTERSECT Self-intersections of a curve.
%
%    [X0,Y0,SEGMENTS] = SELFINTERSECT(X,Y) computes the locations where
%    a curve self-intersects in a fast and robust way.
%    The curve can be broken with NaNs or have vertical segments.
%    Segments of the curve involved in each of the self-interesections are
%    also provided.
%
%    Vectors X and Y are equal-length vectors of at least four points defining
%    the curve.
%    X0 and Y0 are column vectors with the x- and y- coordinates, respectively
%    of the N self-intersections.
%    SEGMENTS is an N x 2 matrix containing the pairs of segments involved in
%    each self-intersection.
%
%    This program uses the theory of operation of the file "Fast and Robust Curve
%    Intersections" submitted by Douglas M. Schwartz (intersections.m, F.Id: 11837).
%
%    Example of use
%  N=201;
%  th=linspace(-3*pi,4*pi,N);
%  R=1;
%  x=R*cos(th)+linspace(0,6,N);
%  y=R*sin(th)+linspace(0,1,N);
%    t0=clock;
%    [x0,y0,segments]=selfintersect(x,y)
%  etime(clock,t0)
%    plot(x,y,'b',x0,y0,'.r');
%  axis ('equal'); grid
%
%    See also INTERSECTIONS.
%
%Version: 1.0, December 11, 2006
%Tested under MATLAB 6.5.0. R13.
%
% (c) Antoni J. Canos.
% ITACA. Techincal University of Valencia (Spain)
% Email:   ancama2@dcom.upv.es
    
    % Input checks.
    % error(nargchk(2,2,nargin))
    % % x and y must be vectors with same number of points (at least 4 for self-intersection).
    % if sum(size(x) > 3) ~= 1 || sum(size(y) > 3) ~= 1 || ...
    %         length(x) ~= length(y)
    %     error('X and Y must be equal-length vectors of at least 4 points.')
    % end
    x0=[];
    y0=[];
    segments=[];
    % Two similar curves are firstly created.
    x1=x; x2=x;
    y1=y; y2=y;
    x1 = x1(:);
    y1 = y1(:);
    x2 = x2(:);
    y2 = y2(:);
    % Compute number of line segments in each curve and some differences we'll
    % need later.
    n1 = length(x1) - 1;
    n2 = length(x2) - 1;
    dxy1 = diff([x1 y1]);
    dxy2 = diff([x2 y2]);
    % Determine the combinations of i and j where the rectangle enclosing the
    % i'th line segment of curve 1 overlaps with the rectangle enclosing the
    % j'th line segment of curve 2.
    [i1,j1] = find(repmat(min(x1(1:end-1),x1(2:end)),1,n2) <= ...
                 repmat(max(x2(1:end-1),x2(2:end)).',n1,1) & ...
                 repmat(max(x1(1:end-1),x1(2:end)),1,n2) >= ...
                 repmat(min(x2(1:end-1),x2(2:end)).',n1,1) & ...
                 repmat(min(y1(1:end-1),y1(2:end)),1,n2) <= ...
                 repmat(max(y2(1:end-1),y2(2:end)).',n1,1) & ...
                 repmat(max(y1(1:end-1),y1(2:end)),1,n2) >= ...
                 repmat(min(y2(1:end-1),y2(2:end)).',n1,1));
    % Removing coincident and adjacent segments.
    remove=find(abs(i1-j1)<2);
    i1(remove)=[];
    j1(remove)=[];
    % Removing duplicate combinations of segments.
    remove=[];
    
    for ii=1:size(i1,1)
        ind=find((i1(ii)==j1(ii:end))&(j1(ii)==i1(ii:end)));
        remove=[remove;ii-1+ind];
    end
    i1(remove)=[];
    j1(remove)=[];
    
    % Find segments pairs which have at least one vertex = NaN and remove them.
    % This line is a fast way of finding such segment pairs.  We take
    % advantage of the fact that NaNs propagate through calculations, in
    % particular subtraction (in the calculation of dxy1 and dxy2, which we
    % need anyway) and addition.
    remove = isnan(sum(dxy1(i1,:) + dxy2(j1,:),2));
    i1(remove) = [];
    j1(remove) = [];
    
    % Find segments pairs which have at least one vertex = NaN and remove them.
    % This line is a fast way of finding such segment pairs.  We take
    % advantage of the fact that NaNs propagate through calculations, in
    % particular subtraction (in the calculation of dxy1 and dxy2, which we
    % need anyway) and addition.
    remove = isnan(sum(dxy1(i1,:) + dxy2(j1,:),2));
    i1(remove) = [];
    j1(remove) = [];
    
    % Initialize matrices.  We'll put the T's and B's in matrices and use them
    % one column at a time.  For some reason, the \ operation below is faster
    % on my machine when A is sparse so we'll initialize a sparse matrix with
    % the fixed values and then assign the changing values in the loop.
    n = length(i1);
    T = zeros(4,n);
    A = sparse([1 2 3 4],[3 3 4 4],-1,4,4,8);
    B = -[x1(i1) x2(j1) y1(i1) y2(j1)].';
    index_dxy1 = [1 3];  %  A(1) = A(1,1), A(3) = A(3,1)
    index_dxy2 = [6 8];  %  A(6) = A(2,2), A(8) = A(4,2)
                         % Loop through possibilities.  Set warning not to trigger for anomalous
                         % results (i.e., when A is singular).
    warning_state = warning('off','MATLAB:singularMatrix');
    try
        for k = 1:n
            A(index_dxy1) = dxy1(i1(k),:);
            A(index_dxy2) = dxy2(j1(k),:);
            T(:,k) = A\B(:,k);
        end
        warning(warning_state)
    catch
        warning(warning_state)
        rethrow(lasterror)
    end
    
    % Find where t1 and t2 are between 0 and 1 and return the corresponding x0
    % and y0 values.  Anomalous segment pairs can be segment pairs that are
    % colinear (overlap) or the result of segments that are degenerate (end
    % points the same).  The algorithm will return an intersection point that
    % is at the center of the overlapping region.  Because of the finite
    % precision of floating point arithmetic it is difficult to predict when
    % two line segments will be considered to overlap exactly or even intersect
    % at an end point.  For this algorithm, an anomaly is detected when any
    % element of the solution (a single column of T) is a NaN.
    in_range = T(1,:) >= 0 & T(2,:) >= 0 & T(1,:) < 1 & T(2,:) < 1;
    anomalous = any(isnan(T));
    if any(anomalous)
        ia = i1(anomalous);
        ja = j1(anomalous);
        % set x0 and y0 to middle of overlapping region.
        T(3,anomalous) = (max(min(x1(ia),x1(ia+1)),min(x2(ja),x2(ja+1))) + ...
                          min(max(x1(ia),x1(ia+1)),max(x2(ja),x2(ja+1))))/2;
        T(4,anomalous) = (max(min(y1(ia),y1(ia+1)),min(y2(ja),y2(ja+1))) + ...
                          min(max(y1(ia),y1(ia+1)),max(y2(ja),y2(ja+1))))/2;
        x0 = T(3,in_range | anomalous).';
        y0 = T(4,in_range | anomalous).';
        i1=i1(in_range | anomalous);
        j1=j1(in_range | anomalous);
    else
        x0 = T(3,in_range).';
        y0 = T(4,in_range).';
        i1=i1(in_range);
        j1=j1(in_range);
    end
    segments=sort([i1,j1],2);
    if isempty (segments)
        i_out = []; j_out = [];
    else
        i_out = segments(:,1);
        j_out = segments(:,2);

    end
end

function [xCurve1,yCurve1]=projectCurve(xCurve1,yCurve1,xCurve2,yCurve2)
    if isempty(xCurve1) || length(xCurve2)<2, return; end
    for i=1:length(xCurve1)
        [dist,pn] = point2linedist(xCurve2,yCurve2,xCurve1(i),yCurve1(i));
        [tmp,j] = min(dist);
        xCurve1(i) = (xCurve1(i)+pn(j,1))/2;
        yCurve1(i) = (yCurve1(i)+pn(j,2))/2;
    end
end

function [dist,pn] = point2linedist(xline,yline,xp,yp)
% point2linedist: distance,projections(line,point).
% A modification of SEGMENT_POINT_DIST_2D
% (http://people.scs.fsu.edu/~burkardt/m_src/geometry/segment_point_dist_2d.m)
    dist = zeros(length(xline)-1,1);
    pn = zeros(length(xline)-1,2);
    p = [xp,yp];
    for i=2:length(xline)
        p1 = [xline(i-1) yline(i-1)];
        p2 = [xline(i) yline(i)];
        if isequal(p1,p2)
            t = 0;
        else
            bot = sum((p2-p1).^2);
            t = (p-p1)*(p2-p1)'/bot;
            % if max(max(t))>1 || min(min(t))<0, dist=-1; return; end
            t = max(t,0);
            t = min(t,1);
        end
        pn(i-1,:) = p1 + t * ( p2 - p1 );
        dist(i-1) = sum ( ( pn(i-1,:) - p ).^2 );
    end
end