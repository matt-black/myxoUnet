%% reset approved and masks and run spineLabeler to check cell masks.

data = dataset5;
cellList = data.cellList;
Image = data.l;
Mask = data.mask;
for i = 1:numel(cellList)
    cellList(i).approved = false;
    cellList(i).mask = [];
end

save('data.mat','cellList','Image','Mask');

%%
spineLabeler(Image, Mask);
