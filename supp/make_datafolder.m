clear, clc, close all
%% SETUP
addpath (genpath (fullfile (pwd, 'cellLabeler')));

%% PARAMETERS
FLDR_NAME = 'test';                     % name of folder to make
PCT_TRAIN = 80;                         % percent of data to put in training set (other goes in validation)
CELL_WIDTH = 4;
BOTHAT_STREL = strel ('disk', 5);

%% PROCESS

% get list of mat files to add
matfilez = dir (fullfile (pwd, 'stream_data'));
matfilez(1:2) = [];                     % get rid of '.', '..' entries

% figure out which files go in training/testation set
n_filez = numel (matfilez);
n_train = ceil (n_filez * (PCT_TRAIN/100));
n_test = n_filez - n_train;
test_idx = randi (n_filez, [n_test, 1]);
train_idx = setdiff ((1:n_filez)', test_idx);

% make directory structure
if exist (fullfile (pwd, FLDR_NAME), 'dir')
    rmdir (fullfile (pwd, FLDR_NAME), 's');
end
mkdir (fullfile (pwd, FLDR_NAME));
mkdir (fullfile (pwd, FLDR_NAME, 'train', 'img'));
mkdir (fullfile (pwd, FLDR_NAME, 'train', 'msk'));
mkdir (fullfile (pwd, FLDR_NAME, 'test', 'img'));
mkdir (fullfile (pwd, FLDR_NAME, 'test', 'msk'));

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
    data.cellList(~appr) = [];
    % write uint16 laser file
    img = im2uint16 (data.Image);
    % make masks
    j4m = makeJ4mask (cell_list, BOTHAT_STREL, size (img), CELL_WIDTH, 8);
    j3m = makeJ3mask (cell_list, size (img), CELL_WIDTH, 8);
    cell_lbl = makeCellMask (cell_list, size (img), CELL_WIDTH, true);
    cell_rgb = label2rgb (cell_lbl, 'jet', 'k', 'shuffle');
    if (any (fi == train_idx))
        write_dir = fullfile (pwd, FLDR_NAME, 'train');
        train_dict = vertcat (train_dict, {curr_train_idx, name});
        write_idx = curr_train_idx;
    else
        write_dir = fullfile (pwd, FLDR_NAME, 'test');
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
    imwrite (cell_rgb, fullfile (write_dir, 'msk', ...
                                 sprintf ('im%03d_cells.png', write_idx)));
    % iterate idx
    if (any (fi == train_idx))
        curr_train_idx = curr_train_idx + 1;
    else
        curr_test_idx = curr_test_idx + 1;
    end
end

trainT = cell2table (train_dict, 'VariableNames', {'idx', 'name'});
writetable (trainT, fullfile (pwd, FLDR_NAME, 'train.csv'));
testT = cell2table (test_dict, 'VariableNames', {'idx', 'name'});
writetable (testT, fullfile (pwd, FLDR_NAME, 'test.csv'));
