% Example script for basic Kohonen map algorithm.
clear all
close all
tic

data = dlmread('data.txt'); % read in data
labels = dlmread('labels.txt'); % read in labels

name = 'Chiara Gastaldi'; % REPLACE BY YOUR OWN NAME
% name = 'Gonçalo Vítor';
targetdigits = name2digits(name); % assign the four digits that should be used

digitstoremove = setdiff(0:9,targetdigits); % the other 6 digits are removed from the data set.
for i=1:6
    data(labels==digitstoremove(i),:) = [];
    labels(labels==digitstoremove(i)) = [];
end

dim = 28*28; % dimension of the images
range = 255; % input range of the images ([0, 255])
[dy, dx]=size(data);

% save histogram and prototype visualization?
save = false;

% set the size of the Kohonen map. In this case it will be 6 X 6
sizeK=6;

class = zeros(sizeK^2,1);
Npoints = zeros(sizeK^2,1);
list = zeros(sizeK^2,1);

%set the width of the neighborhood via the width of the gaussian that
%describes it
% sigma0=3;
sigma0=3;
%initialise the centers randomly
centers=rand(sizeK^2,dim)*range;

% build a neighborhood matrix
neighbor = reshape(1:sizeK^2,sizeK,sizeK);

% YOU HAVE TO SET A LEARNING RATE HERE:
% eta0 = .7;
eta0 = .05;
%set the maximal iteration count
tmax=20000; % this might or might not work; use your own convergence criterion

% runoff = 200; % number of iterations that are not taken into account ->
% doesn't help a lot =S
runoff = 1;

%set the random order in which the datapoints should be presented
iR=mod(randperm(tmax),dy)+1;
convergence = 0;
t = 1;

while(convergence==0 & t<tmax)
    i=iR(mod(t,length(iR))+1);
%     [new_centers wu]=som_step(centers,data(i,:),neighbor,eta0/log(t+1),sigma0); % log rule on eta
%     [new_centers wu]=som_step(centers,data(i,:),neighbor,eta0,sigma0/log(t+1)); % log rule on sigma
    [new_centers wu]=som_step(centers,data(i,:),neighbor,eta0,sigma0);
    
%     max(sum((centers-new_centers).^2, 2)) % print val of conv metrics 2
    max(max(abs(centers-new_centers))) % print val of conv metrics 1
    if all(abs(centers-new_centers)<.1)
        convergence = 1;
    end
    h = waitbar(t/tmax);
    if(t~=tmax-1)
        centers = new_centers;
    end
    list(wu, Npoints(wu)+1) = labels(i); % used for deprecated histogram analysis (for classification)
    class(wu) = (class(wu)*Npoints(wu)+labels(i))/(Npoints(wu)+1);
    Npoints(wu) = Npoints(wu) + 1;
    t = t + 1;
%     break;
end
close(h);
toc

%% for visualization, you can use this:
close all

str = strcat('sigma0=',num2str(sigma0),', map size=',num2str(sizeK), ', eta=',num2str(eta0),', tmax=',num2str(tmax));
red_str = strcat('vars',num2str(sigma0),'ms',num2str(sizeK), 'e',num2str(eta0*100),'t',num2str(tmax));
% red_str = 'etacte_sigmavar';
% red_str = 'etacte_sigmacte3';
histogram = figure;
for i=1:sizeK^2
    subplot(sizeK,sizeK,i);
    hist(list(i, list(i,:)>0))
    title({'Prototype ',i});
    xlim([1 9])
end
% ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
% text(0.5, 1,str,'HorizontalAlignment','center','VerticalAlignment', 'top');
if(save)
    saveas(histogram, strcat('histogram',red_str), 'png');
end
classif = figure;
for i=1:sizeK^2
    subplot(sizeK,sizeK,i);
    imagesc(reshape(centers(i,:),28,28)'); colormap gray;
    axis off
    closest = findClosest(centers(i,:), data, labels); % classification method 2
%     title(mode(list(i,find(list(i,runoff:end))))); % classification method 1 (deprecated)
    title(closest);
    ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(0.5, 1,str,'HorizontalAlignment','center','VerticalAlignment', 'top');
end
if(save)
    disp 'saving image';
    saveas(classif, strcat('classif',red_str), 'png');
end
