function [ okay ] = make_datafolder (fldr_name, data_path, pct_train, ...
    cell_width, aug_dim, bothat_rad)

    this_path = mfilename ('fullpath');
    [this_fldr, ~, ~] = fileparts (this_path);
    addpath (genpath (fullfile (this_fldr, 'cellLabeler')))
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
    mkdir (fullfile (pwd, fldr_name, 'test', 'img'));
    mkdir (fullfile (pwd, fldr_name, 'test', 'msk'));

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
        aug_se = strel ('square', aug_dim);
        bothat_se = strel ('disk', bothat_rad);
        [cell_lbl, j3m, j4m, cell_msk] = generateMasks (...
            cell_list, size (img), cell_width, aug_se, bothat_se);
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
        imwrite (j4m, fullfile (write_dir, 'msk', ...
            sprintf ('im%03d_j4.png', write_idx)));
        imwrite (j3m, fullfile (write_dir, 'msk', ...
            sprintf ('im%03d_j3.png', write_idx)));
        imwrite (uint8 (cell_msk), fullfile (write_dir, 'msk', ...
            sprintf ('im%03d_cell.png', write_idx)));
        imwrite (cell_rgb, fullfile (write_dir, 'msk', ...
            sprintf ('im%03d_clbl.png', write_idx)));
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