load ../result/fivetwo_reuters_Result.mat;
mea = meanErrors; mstd = stdErrors;
mea = [meanErrors(:,1:end-2), meanErrors(:,end)];
mstd = [stdErrors(:,1:end-2), stdErrors(:,end)];

rmse = sqrt(mean(mea.^2));
rmseStd = sqrt(mean(mstd.^2));

round(rmse,2)
round(rmseStd,2)

load ../result/fivetwo_image_Result.mat;

mea = meanErrors; mstd = stdErrors;
mea = [meanErrors(:,1:end-2), meanErrors(:,end)];
mstd = [stdErrors(:,1:end-2), stdErrors(:,end)];

rmse = sqrt(mean(mea.^2))
rmseStd = sqrt(mean(mstd.^2))

round(rmse,2)
round(rmseStd,2)