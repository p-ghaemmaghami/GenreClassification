%%MEG
clear all;
load('Dt_GrnCls_GRDFt.mat');
load('Video_Genres_mat.mat');
targets = zeros(size(Video_Genres_mat(:,1)));
targets(Video_Genres_mat(:,1)) = Video_Genres_mat(:,2);
WlchFt(~isfinite(WlchFt)) = 0;
[subjectNum,clipNum,featureNum] = size(WlchFt);
features = [];
for i = 1 : subjectNum
    %features{i} = squeeze(WlchFt(i,:,:));
    features{i} = squeeze(WlchFt(i,:,[1:102,151:252,301:402,451:552]));
    %features{i} = squeeze(WlchFt(i,:,[103:150,253:300,403:450,553:600]));
    features{i} = features{i}(:,sum(features{i})~=0);
    features{i} = (features{i} - mean2(features{i}))/std2(features{i});
    t = mapstd(features{i}');
    features{i} = t';
    disp(i);
end
total = zeros(4,4);
confMat = [];
acc = [];
for i = 1 : subjectNum
    for j = 1 : clipNum
        trainInd = setdiff(1:36,j);
        trainFeatures = features{i}(trainInd,:);
        testFeatures = features{i}(j,:);
        trainTargets = targets(trainInd);
        testTargets = targets(j);
        model = NaiveBayes.fit(trainFeatures,trainTargets);
        testOutputsMEG(j,i) = model.predict(testFeatures);
        
    end
    confMat{i} = confusionmat(targets,testOutputsMEG(:,i));
    total = total + confMat{i};
    acc(i) = sum(diag(confMat{i}))/clipNum;
    disp(i);
end
normalizedTotal = total/sum(sum(total));
MEGACC = sum(diag(total))/sum(sum((total)));

%% MCA
load('Dt_MCA.mat');
MovieFeatures = mapstd(MovieFeatures');
MovieFeatures = MovieFeatures';
testOutputsMCA = [];
for j = 1 : clipNum
    trainInd = setdiff(1:36,j);
    trainFeatures = MovieFeatures(trainInd,:);
    testFeatures = MovieFeatures(j,:);
    trainTargets = targets(trainInd);
    testTargets = targets(j);
    model = NaiveBayes.fit(trainFeatures,trainTargets);
    testOutputsMCA(j) = model.predict(testFeatures);
end
MovieconfMat = confusionmat(targets,testOutputsMCA);
normalizedMovieconfMat = MovieconfMat/sum(sum(MovieconfMat));
MovieACC = sum(diag(MovieconfMat))/sum(sum((MovieconfMat)));

%% Movie + MEG features
total = zeros(4,4);
confMat = [];
acc = [];
for i = 1 : subjectNum
    for j = 1 : clipNum
        trainInd = setdiff(1:36,j);
        trainFeatures = [features{i}(trainInd,:) MovieFeatures(trainInd,:)];
        testFeatures = [features{i}(j,:)  MovieFeatures(j,:)];
        trainTargets = targets(trainInd);
        testTargets = targets(j);
        model = NaiveBayes.fit(trainFeatures,trainTargets);
        testOutputsMEGMCA(j,i) = model.predict(testFeatures);
    end
    confMat{i} = confusionmat(targets,testOutputsMEGMCA(:,i));
    total = total + confMat{i};
    acc(i) = sum(diag(confMat{i}))/clipNum;
    disp(i);
end
normalizedTotalMovieMEG = total/sum(sum(total));
MovieMEGACC = sum(diag(total))/sum(sum((total)));

%% Population analysis
testOutputsMEGMCA = zeros(clipNum,subjectNum);
testOutputsMEG = zeros(clipNum,subjectNum);
for i = 1 : 36
    [~,majorityVoteMEG(i)] = max([sum(testOutputsMEG(i,:)==1),sum(testOutputsMEG(i,:)==2),sum(testOutputsMEG(i,:)==3),sum(testOutputsMEG(i,:)==4)]);
    [~,majorityVoteMEGMCA(i)] = max([sum(testOutputsMEGMCA(i,:)==1),sum(testOutputsMEGMCA(i,:)==2),sum(testOutputsMEGMCA(i,:)==3),sum(testOutputsMEGMCA(i,:)==4)]);
end
populationACCMEGMCA=sum(majorityVoteMEGMCA==targets')/36;
populationACCMEG=sum(majorityVoteMEG==targets')/36;
populationConfusionMEGMCA = confusionmat(targets,majorityVoteMEGMCA);
populationConfusionMEG = confusionmat(targets,majorityVoteMEG);
save('PopulationResults.mat','majorityVoteMEGMCA','majorityVoteMEG','testOutputsMCA');

%% Random
for it = 1 : 100
    for i = 1 : 30
        randFeatures{i} = randn(36,510);
    end
    total = zeros(4,4);
    confMat = [];
    for i = 1 : subjectNum
        for j = 1 : clipNum
            trainInd = setdiff(1:36,j);
            trainFeatures = randFeatures{i}(trainInd,:);
            testFeatures = randFeatures{i}(j,:);
            trainTargets = targets(trainInd);
            testTargets = targets(j);
            model = NaiveBayes.fit(trainFeatures,trainTargets);
            testOutputs(j) = model.predict(testFeatures);
            
        end
        confMat{i} = confusionmat(targets,testOutputs);
        total = total + confMat{i};
        
    end
    randomACC(it) = sum(diag(total))/sum(sum((total)));
    disp(it);
end
barwitherr([0,0,0,0],[mean(randomACC),MEGACC,MovieACC,MovieMEGACC]);
set(gca,'LineWidth',2,'FontSize',12,'XTickLabel',{'Random','MEG','MCA','MEG+MCA'},'XTick',[1,2,3,4]);
ylim([0.2,0.5])
ylabel('Accuracy');
colormap('summer');

%% Confusion Matrix
clim = minmax([normalizedMovieconfMat(:); normalizedTotal(:)]');
figure;
subplot(1,2,1);
imagesc(normalizedMovieconfMat,clim);
colorbar;
title('Multimedia Features','FontSize',12);
set(gca,'LineWidth',2,'FontSize',12,'XTickLabel',{'Comedy','Romantic','Drama','Horror'},'YTickLabel',{'Comedy','Romantic','Drama','Horror'},'XTick',[1,2,3,4],'YTick',[1,2,3,4]);
subplot(1,2,2);
imagesc(normalizedTotal,clim);
title('MEG Features','FontSize',12);
colorbar;
colormap('Gray');
set(gca,'LineWidth',2,'FontSize',12,'XTickLabel',{'Comedy','Romantic','Drama','Horror'},'YTickLabel',{'Comedy','Romantic','Drama','Horror'},'XTick',[1,2,3,4],'YTick',[1,2,3,4]);

[r,p] = corr(normalizedMovieconfMat(:),normalizedTotal(:));