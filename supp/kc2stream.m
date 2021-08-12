%% setup/load
% convert Katie's data to format for stream_data so that it can be
% integrated into my data

clear, clc, close all
load (fullfile (pwd, 'katie_monolayer_08_06_2021.mat'));

%% dataset1
im = uint16 (dataset1.l);
Image = imflatfield (im, 10);
cellList = dataset1.cellList;
Mask = dataset1.cells;

save (fullfile (pwd, 'stream_data', 'katie_monolayer_dataset1.mat'), ...
    'Image', 'cellList', 'Mask');
clearvars Image cellList Mask

%% dataset2
im = uint16 (dataset2.l);
Image = imflatfield (im, 10);
cellList = dataset2.cellList;
Mask = dataset2.cells;

save (fullfile (pwd, 'stream_data', 'katie_monolayer_dataset2.mat'), ...
    'Image', 'cellList', 'Mask');
clearvars Image cellList Mask

%% dataset3
im = uint16 (dataset3.l);
Image = imflatfield (im, 10);
cellList = dataset3.cellList;
Mask = dataset3.mask;

save (fullfile (pwd, 'stream_data', 'katie_monolayer_dataset3.mat'), ...
    'Image', 'cellList', 'Mask');
clearvars Image cellList Mask

%% dataset4
im = uint16 (dataset4.l);
Image = imflatfield (im, 10);
cellList = dataset4.cellList;
Mask = dataset4.mask;

save (fullfile (pwd, 'stream_data', 'katie_monolayer_dataset4.mat'), ...
    'Image', 'cellList', 'Mask');
clearvars Image cellList Mask

%% dataset5
im = uint16 (dataset5.l);
Image = imflatfield (im, 10);
cellList = dataset5.cellList;
Mask = dataset5.mask;

save (fullfile (pwd, 'stream_data', 'katie_monolayer_dataset5.mat'), ...
    'Image', 'cellList', 'Mask');
clearvars Image cellList Mask
