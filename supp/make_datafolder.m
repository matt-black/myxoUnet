clear, clc, close all
%% SETUP
addpath (genpath (fullfile (pwd, 'cellmasker')));

%% PARAMETERS
FLDR_NAME = 'test';                     % name of folder to make
pct_train = 80;                         % percent of data to put in training set (other goes in validation)
CELL_WIDTH = 4;
BOTHAT_STREL = strel ('disk', 5);

%% PROCESS

% get list of mat files to add
matfilez = dir (fullfile (pwd, 'stream_data'));
matfilez(1:2) = [];                     % get rid of '.', '..' entries

% figure out which files go in training/validation set
n_filez = numel (matfilez);
n_train = ceil (n_filez * (pct_train/100));
n_valid = n_filez - n_train;
valid_idx = randi (n_filez, [n_valid, 1]);
train_idx = setdiff ((1:n_filez)', valid_idx);

% make directory structure
if exist (fullfile (pwd, FLDR_NAME), 'dir')
    rmdir (fullfile (pwd, FLDR_NAME), 's');
end
mkdir (fullfile (pwd, FLDR_NAME));
train_dir = fullfile (pwd, FLDR_NAME, 'train'); mkdir (train_dir);
valid_dir = fullfile (pwd, FLDR_NAME, 'valid'); mkdir (valid_dir);
mkdir (fullfile (train_dir, 'img')); mkdir (fullfile (valid_dir, 'img'))
mkdir (fullfile (train_dir, 'msk')); mkdir (fullfile (valid_dir, 'msk'))

% populate directories with data
curr_train_ind = 1; curr_valid_ind = 1;
train_dict = {}; valid_dict = {};
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
    % write to file
    if any (fi == train_idx)
        write_dir = train_dir;
        write_ind = curr_train_ind;
        train_dict = vertcat (train_dict, {curr_train_ind, name});
        curr_train_ind = curr_train_ind + 1;
    else
        write_dir = valid_dir;
        write_ind = curr_valid_ind;
        valid_dict = vertcat (valid_dict, {curr_valid_ind, name});
        curr_valid_ind = curr_valid_ind + 1;
    end
    % write laser file
    imwrite (img, fullfile (write_dir, 'img', ...
                            sprintf ('im%03d.tiff', write_ind)));
    % write masks
    imwrite (j4m, fullfile (write_dir, 'msk', ...
                            sprintf ('im%03d_j4.tiff', write_ind)));
    imwrite (j3m, fullfile (write_dir, 'msk', ...
                            sprintf ('im%03d_j3.tiff', write_ind)));
    imwrite (cell_rgb, fullfile (...
        write_dir, 'msk', sprintf ('im%03d_cells.tiff', write_ind)));    
end

trainT = cell2table (train_dict, 'VariableNames', {'idx', 'name'});
trainT.is_train = true (size (trainT, 1), 1);
validT = cell2table (valid_dict, 'VariableNames', {'idx', 'name'});
validT.is_train = false (size (validT, 1), 1);

T = vertcat (trainT, validT);
writetable (T, fullfile (pwd, FLDR_NAME, 'lookup_table.csv'));
