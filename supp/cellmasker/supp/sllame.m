function sline = sllame(vx,vy,u0,SL,Fx,Fy,donezo,l,th)

    %% Find cell midline through a given point.
    % Inputs:
    % vx, vy    - director field x and y components.
    % u0        - Given point on cell body to start the midline from.
    % Fx, Fy    - Gradient of brightness image, x and y components.
    % donezo    - Mask of regions to exclude from new cells.
    % l         - Brightness image
    % th        - Threshold of brighness for cutting off centerlines.

    sz = size(vx);

    function f = ff(u)
    % Function for calculating interpolated x and y components of
    % director field at point u.

        ux = u(1);
        uy = u(2);

        if (ux<1 || uy<1 || ux>sz(2)-1 || uy>sz(1)-1 || isnan(ux) || isnan(uy))
            f = [NaN NaN];
            return
        end
        if (rem(ux,1)==0 && rem(uy,1)==0)
            vxu = vx(uy,ux);
            vyu = vy(uy,ux);
        else
            if (rem(ux,1)==0)
                ux = ux+0.00001;
            end
            if (rem(uy,1)==0)
                uy = uy+0.00001;
            end
            % Interpolate the director field to follow for next centerline
            % point.
            ux1 = floor(u(1));
            ux2 = ceil(u(1));
            uy1 = floor(u(2));
            uy2 = ceil(u(2));

            vx11 = vx(uy1,ux1);
            vx12 = vx(uy1,ux2);
            vx21 = vx(uy2,ux1);
            vx22 = vx(uy2,ux2);

            vy11 = vy(uy1,ux1);
            vy12 = vy(uy1,ux2);
            vy21 = vy(uy2,ux1);
            vy22 = vy(uy2,ux2);

            if vx11*vx12+vy11*vy12<0
                vx12 = -vx12;
                vy12 = -vy12;
            end
            if vx12*vx22+vy12*vy22<0
                vx22 = -vx22;
                vy22 = -vy22;
            end
            if vx22*vx21+vy22*vy21<0
                vx21 = -vx21;
                vy21 = -vy21;
            end

            vxu = 1/((ux2-ux1)*(uy2-uy1))*[(ux2-ux) (ux-ux1)]*[vx11 vx12; vx21 vx22]...
                  *[uy2-uy; uy-uy1];

            vyu = 1/((ux2-ux1)*(uy2-uy1))*[(ux2-ux) (ux-ux1)]*[vy11 vy12; vy21 vy22]...
                  *[uy2-uy; uy-uy1];
        end
        % Return normalized vector along director field at the current test
        % point.
        vmag = sqrt(vxu^2+vyu^2);
        f = [vxu/vmag, vyu/vmag];

    end

    % Start streamline (u) at input point u0.
    u = u0;
    ls = l(u0(2),u0(1)); % Record brightness along streamline.
    fp = ff(u); % fp is the direction from the previous point (to avoid >90 degree flips).

    for i=1:round(SL/2) % First half of the streamline going in
        f = ff(u(i,:));
        if f(1)*fp(1)+f(2)*fp(2)<0 % If new direction is >90 degrees off the last one flip it.
            f = -f;
        end
        if f(1)*fp(1)+f(2)*fp(2)<cos(pi/4)
            f = [NaN NaN];
        end

        % Next point is current point plus the director field direction.
        newux = u(end,1)+f(1);
        newuy = u(end,2)+f(2);
        try
            % Interpolate x and y components of the image gradient.
            newux1 = floor(newux);
            newux2 = ceil(newux);
            newuy1 = floor(newuy);
            newuy2 = ceil(newuy);

            Fx11 = Fx(newuy1,newux1);
            Fx12 = Fx(newuy1,newux2);
            Fx21 = Fx(newuy2,newux1);
            Fx22 = Fx(newuy2,newux2);

            Fy11 = Fy(newuy1,newux1);
            Fy12 = Fy(newuy1,newux2);
            Fy21 = Fy(newuy2,newux1);
            Fy22 = Fy(newuy2,newux2);

            Fxu = 1/((newux2-newux1)*(newuy2-newuy1))*[(newux2-newux)...
                                (newux-newux1)]*[Fx11 Fx12; Fx21 Fx22]...
                  *[newuy2-newuy; newuy-newuy1];

            Fyu = 1/((newux2-newux1)*(newuy2-newuy1))*[(newux2-newux)...
                                (newux-newux1)]*[Fy11 Fy12; Fy21 Fy22]...
                  *[newuy2-newuy; newuy-newuy1];
        catch
            Fxu = NaN;
            Fyu = NaN;
        end

        if ~isnan(Fyu)
            if donezo(round(newuy+Fyu),round(newux+Fxu))==0
                % If the next point is still in valid unclaimed territory
                % push it along the gradient as record it as the next point
                % on the centerline.

                u = [u; newux+Fxu newuy+Fyu];

                % Interpolate the brightness data at this endpoint.
                newuxa = newux+Fxu;
                newuya = newuy+Fyu;

                newux1 = floor(newuxa);
                newux2 = ceil(newuxa);
                newuy1 = floor(newuya);
                newuy2 = ceil(newuya);

                l11 = l(newuy1,newux1);
                l12 = l(newuy1,newux2);
                l21 = l(newuy2,newux1);
                l22 = l(newuy2,newux2);

                lu = 1/((newux2-newux1)*(newuy2-newuy1))*[(newux2-newux)...
                                    (newux-newux1)]*[l11 l12; l21 l22]...
                     *[newuy2-newuy; newuy-newuy1];
                ls = [ls; l(round(newuya),round(newuxa))];

                %ls = [ls; lu];
                fp = f;

                % If the brightness is below the threshold end the
                % centerline here.
                if ls(end)<th
                    break;
                end

            else
                break;
            end
        else
            break;
        end
    end

    % Start again at the test point (u0) and find the center line in the
    % opposite direction.

    fp = -ff([u(1,1) u(1,2)]);
    for i=1:round(SL/2)
        f = ff(u(1,:));
        if f(1)*fp(1)+f(2)*fp(2)<0
            f = -f;
        end
        if f(1)*fp(1)+f(2)*fp(2)<cos(pi/4)
            f = [NaN NaN];
        end
        newux = u(1,1)+f(1);
        newuy = u(1,2)+f(2);
        try
            newux1 = floor(newux);
            newux2 = ceil(newux);
            newuy1 = floor(newuy);
            newuy2 = ceil(newuy);

            Fx11 = Fx(newuy1,newux1);
            Fx12 = Fx(newuy1,newux2);
            Fx21 = Fx(newuy2,newux1);
            Fx22 = Fx(newuy2,newux2);

            Fy11 = Fy(newuy1,newux1);
            Fy12 = Fy(newuy1,newux2);
            Fy21 = Fy(newuy2,newux1);
            Fy22 = Fy(newuy2,newux2);

            Fxu = 1/((newux2-newux1)*(newuy2-newuy1))*[(newux2-newux)...
                                (newux-newux1)]*[Fx11 Fx12; Fx21 Fx22]...
                  *[newuy2-newuy; newuy-newuy1];

            Fyu = 1/((newux2-newux1)*(newuy2-newuy1))*[(newux2-newux)...
                                (newux-newux1)]*[Fy11 Fy12; Fy21 Fy22]...
                  *[newuy2-newuy; newuy-newuy1];
        catch
            Fxu = NaN;
            Fyu = NaN;
        end
        if ~isnan(Fyu)
            try
                if donezo(round(newuy+Fyu),round(newux+Fxu))==0
                    u = [newux+Fxu newuy+Fyu; u];

                    newuxa = newux+Fxu;
                    newuya = newuy+Fyu;

                    newux1 = floor(newuxa);
                    newux2 = ceil(newuxa);
                    newuy1 = floor(newuya);
                    newuy2 = ceil(newuya);

                    l11 = l(newuy1,newux1);
                    l12 = l(newuy1,newux2);
                    l21 = l(newuy2,newux1);
                    l22 = l(newuy2,newux2);

                    lu = 1/((newux2-newux1)*(newuy2-newuy1))*[(newux2-newux)...
                                        (newux-newux1)]*[l11 l12; l21 l22]...
                         *[newuy2-newuy; newuy-newuy1];

                    ls = [ls; lu];
                    fp = f;

                    if ls(end)<th
                        break;
                    end
                else
                    break;
                end
            catch
                keyboard;
            end
        else
            break;
        end
    end

    % Return the calculated centerline.
    sline = u;
end