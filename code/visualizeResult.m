% %% Visualize of the Results as Graph
% % This script visualizes the mean errors and standard deviation over
% % Reuters and image datasets for cross-validation sampling results and
% % the average results as graphs.
% 
% load ../result/average_image_Result.mat;
% mea = meanErrors'; mstd = stdErrors';
% figure;
% errorbar(mea(end,:),mstd(end,:))
% hold on;
% errorbar(mea(end-1,:),mstd(end-1,:))
% errorbar(mea(3,:),mstd(3,:));
% errorbar(mea(4,:),mstd(4,:))
% errorbar(mea(5,:),mstd(5,:))
% ylabel('Error in %')
% xlabel('No. Image Dataset')
% legend('PCTKVM','PCTKVMest','TCA','JDA','GFK');
% xlim([0 13])
% hold off;
% 
% figure;
% errorbar(mea(end,:),mstd(end,:))
% hold on;
% errorbar(mea(end-1,:),mstd(end-1,:))
% errorbar(mea(2,:),mstd(2,:));
% errorbar(mea(1,:),mstd(1,:))
% errorbar(mea(6,:),mstd(6,:))
% ylabel('Error in %')
% xlabel('No. Image Dataset')
% legend('PCTKVM','PCTKVMest','PCVM','SVM','TKL');
% xlim([0 13])
% hold off;
% %-------
% clear all;
% load ../result/average_reuters_Result.mat;
% mea = meanErrors'; mstd = stdErrors';
% figure;
% errorbar(mea(end,:),mstd(end,:))
% hold on;
% errorbar(mea(end-1,:),mstd(end-1,:))
% errorbar(mea(3,:),mstd(3,:));
% errorbar(mea(4,:),mstd(4,:))
% errorbar(mea(5,:),mstd(5,:))
% ylabel('Error in %')
% xlabel('No. Reuters Dataset')
% legend('PCTKVM','PCTKVMest','TCA','JDA','GFK');
% xlim([0 7])
% 
% figure;
% errorbar(mea(end,:),mstd(end,:))
% hold on;
% errorbar(mea(end-1,:),mstd(end-1,:))
% errorbar(mea(2,:),mstd(2,:));
% errorbar(mea(1,:),mstd(1,:))
% errorbar(mea(6,:),mstd(6,:))
% ylabel('Error in %')
% xlabel('No. Reuters Dataset')
% legend('PCTKVM','PCTKVMest','PCVM','SVM','TKL');
% xlim([0 7])
% hold off;
%----
% clear all;
% 
% load ../result/fivetwo_image_Result.mat;
% mea = meanErrors'; mstd = stdErrors';
% figure;
% errorbar(mea(end,:),mstd(end,:))
% hold on;
% errorbar(mea(end-1,:),mstd(end-1,:))
% errorbar(mea(3,:),mstd(3,:));
% errorbar(mea(4,:),mstd(4,:))
% errorbar(mea(5,:),mstd(5,:))
% ylabel('Error in %')
% xlabel('No. Image Dataset')
% legend('PCTKVM','PCTKVMest','TCA','JDA','GFK');
% xlim([0 13])
% hold off;
% 
% load ../result/fivetwo_image_Result.mat;
% mea = meanErrors'; mstd = stdErrors';
% figure;
% errorbar(mea(end,:),mstd(end,:))
% hold on;
% errorbar(mea(end-1,:),mstd(end-1,:))
% errorbar(mea(2,:),mstd(2,:));
% ylabel('Error in %')
% xlabel('No. Image Dataset')
% legend('PCTKVM','PCTKVMest','PCVM');
% xlim([0 13])
% hold off;
% 
% load ../result/fivetwo_reuters_Result.mat;
% mea = meanErrors'; mstd = stdErrors';
% figure;
% errorbar(mea(end,:),mstd(end,:))
% hold on;
% errorbar(mea(end-1,:),mstd(end-1,:))
% errorbar(mea(3,:),mstd(3,:));
% errorbar(mea(4,:),mstd(4,:))
% errorbar(mea(5,:),mstd(5,:))
% ylabel('Error in %')
% xlabel('No. Reuters Dataset')
% legend('PCTKVM','PCTKVMest','TCA','JDA','GFK');
% xlim([0 7])
% hold off;

load ../result/fivetwo_reuters_Result.mat;
mea = meanErrors'; mstd = stdErrors';
figure;
errorbar(mea(end,:),mstd(end,:))
hold on;
errorbar(mea(end-1,:),mstd(end-1,:))
errorbar(mea(2,:),mstd(2,:));
ylabel('Error in %')
xlabel('No. Reuters Dataset')
legend('PCTKVM','PCTKVMest','PCVM');
xlim([0 7])
hold off;