function [ okay ] = make_dunet_datafolder (fldr_name, data_path, pct_train, ...
                                           cell_width, neig_strel, save_rgb_labels)
%MAKE_DATAFOLDER
    this_path = mfilename ('fullpath');
    [this_fldr, ~, ~] = fileparts (this_path);
    addpath (genpath (fullfile (this_fldr, 'cellLabeler')))
    %% ARGUMENT PROCESSING
    
    narginchk (3, 6)
    if nargin < 6, save_rgb_labels = true;
        if nargin < 5, neig_strel = strel ('disk', 3); 
            if nargin < 4, cell_width = 3; end
        end
    end
    
    %% PROCESS
    % get list of mat files to add
    matfilez = dir (data_path);
    matfilez(1:2) = [];                     % get rid of '.', '..' entries

    % figure out which files go in training/testation set
    n_filez = numel (matfilez);
    n_train = ceil (n_filez * (pct_train/100));
    n_test = n_filez - n_train;
    test_idx = randi (n_filez, [n_test, 1]);
    train_idx = setdiff ((1:n_filez)', test_idx);

    % make directory structure
    if exist (fullfile (pwd, fldr_name), 'dir')
        rmdir (fullfile (pwd, fldr_name), 's');
    end
    mkdir (fullfile (pwd, fldr_name));
    mkdir (fullfile (pwd, fldr_name, 'train', 'img'));
    mkdir (fullfile (pwd, fldr_name, 'train', 'msk'));
    mkdir (fullfile (pwd, fldr_name, 'train', 'dst'));
    mkdir (fullfile (pwd, fldr_name, 'test', 'img'));
    mkdir (fullfile (pwd, fldr_name, 'test', 'msk'));
    mkdir (fullfile (pwd, fldr_name, 'test', 'dst'));

    train_dict = {}; test_dict = {};
    curr_train_idx = 1; curr_test_idx = 1;
    % populate directories with data
    for fi = 1:n_filez
        data = load (fullfile (matfilez(fi).folder, matfilez(fi).name));
        [~, name, ~] = fileparts (fullfile (...
            matfilez(fi).folder, matfilez(fi).name));
        % filter down to approved cells
        appr = arrayfun (@(x) x.approved, data.cellList);
        cell_list = data.cellList(appr);
        % write uint16 laser file
        img = im2uint16 (data.Image);
        % make masks
        [cell_dist, neig_dist, cell_lbl] = generateDistImgs (...
            cell_list, size(img), cell_width, neig_strel);
        
        cell_rgb = label2rgb (cell_lbl, 'jet', 'k', 'shuffle');
        if (any (fi == train_idx))
            write_dir = fullfile (pwd, fldr_name, 'train');
            train_dict = vertcat (train_dict, {curr_train_idx, name});
            write_idx = curr_train_idx;
        else
            write_dir = fullfile (pwd, fldr_name, 'test');
            test_dict = vertcat (test_dict, {curr_test_idx, name});
            write_idx = curr_test_idx;
        end
        % write laser file
        imwrite (img, fullfile (write_dir, 'img', ...
            sprintf ('im%03d.png', write_idx)));
        % write masks
        save (fullfile (write_dir, 'dst', sprintf ('im%03d.mat', write_idx)), ...
              'cell_dist', 'neig_dist');
        if save_rgb_labels  % save (not-quantitative) rgb labels
            imwrite (cell_rgb, fullfile (write_dir, 'msk', ...
                sprintf ('im%03d_clbl.png', write_idx)));
        else  % image preserves labels
            imwrite (uint16(cell_lbl), fullfile (write_dir, 'msk', ...
                sprintf ('im%03d_clbl.png', write_idx)));
        end
        % iterate idx
        if (any (fi == train_idx))
            curr_train_idx = curr_train_idx + 1;
        else
            curr_test_idx = curr_test_idx + 1;
        end
    end

    trainT = cell2table (train_dict, 'VariableNames', {'idx', 'name'});
    writetable (trainT, fullfile (pwd, fldr_name, 'train.csv'));
    testT = cell2table (test_dict, 'VariableNames', {'idx', 'name'});
    writetable (testT, fullfile (pwd, fldr_name, 'test.csv'));
    okay = true;
end