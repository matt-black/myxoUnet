function [ okay ] = make_kcdf ( fldr_name, data_path, pct_train, ...
                                cell_width, aug_dim, bothat_rad, ...
                                save_rgb_labels )
%MAKE_KCDF generate a dataset for nn training from Katie-formatted data

    this_path = mfilename ('fullpath');
    [this_fldr, ~, ~] = fileparts (this_path);
    addpath (genpath (fullfile (this_fldr, 'cellLabeler')))
    %% ARGUMENT PROCESSING
    narginchk (4, 7)
    if nargin < 7, save_rgb_labels = true;
        if nargin < 6, bothat_rad = 8;
            if nargin < 5, aug_dim = 2; end
        end
    end
    %% PROCESS
    % get # of images in dataset
    mat_obj = matfile (data_path);
    n_img = length (who (mat_obj));

    % figure out train/test sets
    n_train = ceil (n_img * (pct_train/100));
    n_test = n_img - n_train;
    test_idx = randi (n_img, [n_test, 1]);
    train_idx = setdiff ((1:n_img)', test_idx);
    
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
    for fi = 1:n_img
        data = load (data_path, sprintf ('dataset%d', fi));
        data = data.(sprintf ('dataset%d', fi));
        % filter down to approved cells
        appr = arrayfun (@(x) x.approved, data.cellList);
        cell_list = data.cellList(appr);
        % write uint16 laser file
        img = uint16 (data.l);
        % make masks
        aug_se = strel ('square', aug_dim);
        bothat_se = strel ('disk', bothat_rad);
        [cell_lbl, j3m, j4m, cell_msk] = generateMasks (...
            cell_list, size (img), cell_width, aug_se, bothat_se);
        cell_rgb = label2rgb (cell_lbl, 'jet', 'k', 'shuffle');
        if (any (fi == train_idx))
            write_dir = fullfile (pwd, fldr_name, 'train');
            train_dict = vertcat (train_dict, {curr_train_idx, fi});
            write_idx = curr_train_idx;
        else
            write_dir = fullfile (pwd, fldr_name, 'test');
            test_dict = vertcat (test_dict, {curr_test_idx, fi});
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

    trainT = cell2table (train_dict, 'VariableNames', {'idx', 'dataset_number'});
    writetable (trainT, fullfile (pwd, fldr_name, 'train.csv'));
    testT = cell2table (test_dict, 'VariableNames', {'idx', 'dataset_number'});
    writetable (testT, fullfile (pwd, fldr_name, 'test.csv'));
    okay = true;
end