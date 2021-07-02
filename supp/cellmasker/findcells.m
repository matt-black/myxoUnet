function cells = findcells(l, maskout, sm, maxl, minl, th)
%% Find the cells in a certain image. 
    % Inputs:   l - Brightness image (Normalize to be between 0 and 1).
    %           maskout - Mask of regions to check for cells. (Checks
    %           regions with 0s excludes regions with 1s).
    %           sm - Smoothness for calculating director field. (I use 2).
    %           maxl - Maximum cell length in pixels (ish). (I use 80).
    %           minl - Minimum cell length (hard cutoff). (I use 10).
    %           th - Brightthness threshold for cell ends (I use 0.3).
    
    tstrt = tic;

    % Find the director field and image gradients for making strealines

    df = dfield(l,sm);  % Director field function.
    dfx = cos(df);
    dfy = sin(df);
    [grx,gry] = gradient(l);
    
    % Find bright spots to start streamlines at.
    testpts = imregionalmax(l,8);
    
    % Define regions to exclude from cell protection (at least include
    % outer few pixels.)
    edges = zeros(size(l));

    edges(1:3,:) = 1;
    edges(end-2:end,:) = 1;
    edges(:,1:3) = 1;
    edges(:,end-2:end) = 1;
    
%     lays = loaddata(fpath,t,'covid_layers','int8');
%     holes = lays==0;
%     edges(imdilate(holes,strel('disk',1)))=1;
%     lay2s = lays==2;
%     edges(imdilate(lay2s,strel('disk',11)))=1;

    edges(maskout==1)=1;
    testpts(edges==1)=0;

    % List of indices for points to test and convert to x,y coords (u).
    uinds = find(testpts);

    [I, J] = ind2sub(size(l),uinds);
    u = [J I];
    
    is = 1:numel(u(:,1)); % List of points still needed to be tested.
    cells = {}; 
    
    donezo = edges; % Array of points already accounted for in cells or 
                    % masked out areas.
    
    % Some timer variables for code timing.
    tl = 0;
    tr = 0;
    
    while ~isempty(is)
        % Pick a random point on a cell body to find a cell through.
        i = randi(numel(is));
        tic
        
        % Calculate a potential cell center line through point i.
        % Streamline function.
        
        sline = sllame(dfx,dfy,u(is(i),:),2*maxl,grx,gry,donezo,l,th);
        
        sline = sline(2:end-1,:);
        
        tl = tl+toc;
        tic
        
        del = zeros(size(is));

        if numel(sline(:,1))>minl % Use cells which are more minl long.
            cell = sline;

            % Create a list of indices for a mask of the cell.
            
            inds = sub2ind(size(edges),round(sline(:,2)),round(sline(:,1)));
            inds = [inds; sub2ind(size(edges),round(sline(:,2))+1,round(sline(:,1)))];
            inds = [inds; sub2ind(size(edges),round(sline(:,2))-1,round(sline(:,1)))];
            inds = [inds; sub2ind(size(edges),round(sline(:,2)),round(sline(:,1))+1)];
            inds = [inds; sub2ind(size(edges),round(sline(:,2)),round(sline(:,1))-1)];
            
            cells{end+1}.pix = cell;              

            donezo(inds) = 1;
            % Find points to remove from list of potential test cell point.
            del(donezo(uinds(is))==1) = 1;

            del(i) = 1;
            
        else
            del(i) = 1;
        end
        
        is(del==1) = []; % Remove accounted for test points from the list of pts to check.
        tr = tr+toc;

    end

    cells = cell2mat (cells);           % convert to struct array

    toc(tstrt)
    
%     for i = 1:numel(cells)
%         cells{i}.fpath = fpath;
%         cells{i}.t = t;
%     end
%     
end

