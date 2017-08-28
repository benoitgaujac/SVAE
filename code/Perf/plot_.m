formatSpec = '%f%f%f';
perfs_1L    = readtable('full.csv','Delimiter',';','Format',formatSpec);
perfs_2L    = readtable('full2.csv','Delimiter',';','Format',formatSpec);

one = ones(100,1);

f1 = figure
plot(perfs_1L{:,1},perfs_1L{:,2},'--b',perfs_1L{:,1},perfs_1L{:,3},'--b','LineWidth',.9)
%ylim([-340 -80])
%legend('train', 'test','location','southeast')
xlabel('epochs') % x-axis label
ylabel('objective L') % y-axis label
hold on;
plot(perfs_2L{:,1},perfs_2L{:,2},'--r',perfs_2L{:,1},perfs_2L{:,3},'--r','LineWidth',.9)
hold on;
plot(perfs_1L{:,1},max(perfs_2L{:,3})*one,'b','LineWidth',.9)
