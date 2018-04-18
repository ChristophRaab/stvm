load ../result/fivetwo_reuters_Result.mat;
mea = meanErrors; mstd = stdErrors;
mea = [meanErrors(:,1:end-2), meanErrors(:,end)];
mstd = [stdErrors(:,1:end-2), stdErrors(:,end)];


figure;
errorbar(mea,mstd);
hold on;
ylabel('Error in %')
xlabel('No. Reuters Dataset')
legend('SVM','PCVM','TCA','JDA','GFK','TKL','PCTKVM');
xlim([0 7])
hold off;


load ../result/fivetwo_image_Result.mat;
mea = meanErrors; mstd = stdErrors;
mea = [meanErrors(:,1:end-2), meanErrors(:,end)];
mstd = [stdErrors(:,1:end-2), stdErrors(:,end)];
figure;
errorbar(mea(1:end-1,:),mstd(1:end-1,:))
hold on;
ylabel('Error in %')
xlabel('No. Image Dataset');
legend('SVM','PCVM','TCA','JDA','GFK','TKL','PCTKVM');
xlim([0 13])
hold off;