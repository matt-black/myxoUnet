function [ smoothed ] = smoothContour (contour, fsmooth)
%SMOOTHCONTOUR smooth input contour
    if nargin == 1, fsmooth = []; end
    if fsmooth < 0
        smoothed = contour; 
        return
    end
    if (isempty (fsmooth) || fsmooth == 0)
        fsmooth = size (contour, 1) / 2;
    end
    smoothed = ifdescp (frdescp (contour), fsmooth);
end

function [fft_sig] = frdescp (signal)
%FRDESCP fourier descriptors of input Nx2 signal (the boundary)
%
% NOTE: ripped from oufti
    [np, nc] = size(signal);
    if nc ~=2, error('S must be of size np-by-2.'); end
    if np/2 ~= round(np/2)
        signal(end+1,:) = signal(end, :);
        np = np + 1;
    end
    x = 0:(np-1);
    m = ((-1).^x)';
    signal(:,1) = m .* signal(:,1);
    signal(:,2) = m .* signal(:,2);
    signal = signal(:,1) + sqrt(-1)*signal(:,2);
    fft_sig = fft(signal);
end

function [sig] = ifdescp(isig, fsmooth)
%IFDESCP inverse transform of frdescp
%
% NOTE: ripped from oufti
    nsig = length(isig);
    if nargin == 1 || fsmooth>nsig
        fsmooth = nsig;
    end
    x = 0:(nsig-1);
    m = ((-1).^x)';
    d = round((nsig - fsmooth)/2);
    isig(1:d) = 0;
    isig(nsig-d+1:nsig) = 0;
    zz = ifft(isig);
    sig(:,1) = real(zz);
    sig(:,2) = imag(zz);
    sig(:,1) = m.*sig(:,1);
    sig(:,2) = m.*sig(:,2);
end