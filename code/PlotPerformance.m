load reuters1.mat;
mea = meanErrors; mstd = stdErrors;
mea = meanErrors(:,1:end-2);
mea = [mea(:,1:4),mea(:,5)];
mstd = [mstd(:,1:4),mstd(:,5)];

load reuters2.mat;

mea = [mea, meanErrors];
mstd = [mstd,stdErrors];
figure;
errorbar(mea,mstd);
hold on;
ylabel('Error in %')
xlabel('No. Reuters Dataset')
legend('SVM','PCVM','TCA','JDA','TKL','STVM');
xlim([0 7])
print("STVM_Performance_Reuters","-depsc","-r1000")
hold off;


load newsgroup1.mat
mea = meanErrors; mstd = stdErrors;
load newsgroup2.mat


mea = [mea(:,1),meanErrors, mea(:,2:end)];
mstd = [mstd(:,1),stdErrors, mstd(:,2:end)];
figure;
errorbar(mea,mstd);
hold on;
ylabel('Error in %')
xlabel('No. Newsgroup Dataset');
legend('SVM','PCVM','TCA','JDA','TKL','STVM');
xlim([0 7])
print("STVM_Performance_Newsgroup","-depsc","-r1000")
hold off;