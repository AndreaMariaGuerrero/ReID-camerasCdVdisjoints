%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             author : Guerrero Andréa, stagiaire LAAS/CNRS               %
%             subject : apply NCR_generalized method on CAMNET            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Ajouter Gurobi : aller dans gurobi650/linux64/matlab et taper
% gurobi_setup

% clear;
% close all;
% clc;

%% %% Consistent Re-ID in a camera network %% %%

% Author:    Abir Das and Anirban Chakraborty
% Date:      2014/11/25 15:23
% Revision:  1.0
% Copyright: Abir Das, Anirban Chakraborty and Amit K. RoyChowdhury, 2014

%% Parameter initialization and addition of paths

% addpath('NCR_Code-master/Pairwise_Similarity');
% addpath('NCR_Code-master/Results');
% addpath('NCR_Code-master/Results/GeneralizedNCR');
% addpath('Gurobi_test/');

% Dataset name. This will be used to load pairwise wise similarity scores
% and to store intermediate variable and results. For example the pairwise
% similarity score will read from a file named -
% ['Pairwise_sim_Less_People_' dataset_name '.mat']; The optimization
% variables (intermediate) will be stored as ['OptVar_' dataset_name
% '.mat'] etc.
dataset_name = 'camnet'; 
% Number of trials
testset = 1%:10;
% 1:3; 
% Camera numbers for this dataset
cameras = [1 2 3 4];% [1 2 3];

% Number of persons present in each camera. 
numPersons = [11 13 9 14]; %[11 13 9];%[8 9 8]; %[8, 9, 7]; cas (3) %[7, 9, 4]; cas (1) et (2) %%%%%%%%% !!!!!!!
%[20, 20, 16, 16];

% kk is the different 'k' values to try
kk =  0.2:0.05:0.4;%0.2:0.025:0.3;
%% Load pairwise similarity scores and generate optimization cost function and constraints

% Load the pairwise similarity matrices. If the number of cameras be C,
% then this mat file will contain a cell array (named 'pairwise_sim') of
% size CxC, where diagonal as well as the lower diagonal elements of
% 'pairwise_sim' will be empty matrices. Other elements of 'pairwise_sim'
% contains the pairwise similarity scores between the persons in the
% corresponding cameras.

load('pairwise_sim_synchro_4cam_1test.mat');%pairwise_sim_camnet (1) ou pairwise_sim_camnet_rearrange (2) ou pairwise_sim_camnet_autredata (3) pairwise_sim_camnet_corr.mat (4)
%load('NCR_Code-master/Pairwise_Similarity/Pairwise_sim_Less_People_RAiD_6a.mat')

% Track data : pairwise
load('pairwise_4cam_trackinfo.mat');

%pairwise_sim = pairwise_sim_synchro;

% Get the different camera pairs
CPairs = combnk(cameras,2);
numCPairs = size(combnk(cameras,2), 1);

% Number of persons present in both camera of the pair /!\ CAREFUL CPAIRS ORDER !! 
numPers_CPair = [8 9 7 9 7 8]; %[8 7 7];%[8 7 8]; % only use to evaluate %%%%%%%%% !!!!!!!

% Number of cameras
numCameras = length(cameras);

% t_min transition per cameras pairs in frame number
transitions = [350 850 250 175 1 450];



vectorSize_perpair = zeros(numCPairs, 1);
for i = 1:numCPairs
    n1 = CPairs(i,1);
    n2 = CPairs(i,2);
    vectorSize_perpair(i) = numPersons(n1)*numPersons(n2);
end

%% For different k values
for kCount = 1:length(kk)
    change = []; ind_change = [];
    % Compute the cost function
    f = cell(length(testset),1);
    for iTSCount=testset
        for i = 1:numCPairs
            n1 = CPairs(i,1);
            n2 = CPairs(i,2);
            
%             if (numPersons(n1) < numPersons(n2))
%                 numP1 = numPersons(n2);
%                 numP2 = numPersons(n1);
%                 
%                 % We keep pair that need permutation
%                 change = [change; n1 n2];
%                 ind_change = [ind_change; find(CPairs(:,1)==n1 & CPairs(:,2)==n2)];
%                 
%                 clear cc;
%         %         cc = permute(pairwise_sim{n1,n2}(1:numP2,1:numP1,iTSCount), [2 1 3]);
%                 cc = zeros(size(permute(pairwise_sim{n1,n2}(1:numP2,1:numP1,iTSCount), [2 1 3])));
%                 sub_cc = cc(1:numPers_CPair(i),1:numPers_CPair(i));
%                 sub_cc(logical(eye(size(sub_cc)))) = 1;
%                 cc(1:numPers_CPair(i),1:numPers_CPair(i)) = sub_cc;
%               %%%  disp('change');
%             else
                numP1 = numPersons(n1);
                numP2 = numPersons(n2);
                
                clear cc;
                cc =  pairwise_sim{n1,n2}(1:numP1,1:numP2,iTSCount);
%                 cc = zeros(size(pairwise_sim{n1,n2}(1:numP1,1:numP2,iTSCount)));
%                 sub_cc = cc(1:numPers_CPair(i),1:numPers_CPair(i));
%                 sub_cc(logical(eye(size(sub_cc)))) = 1;
%                 cc(1:numPers_CPair(i),1:numPers_CPair(i)) = sub_cc;
%              end
            
            f{iTSCount} = [f{iTSCount}; cc(:)];
        end;
        f{iTSCount} = kk(kCount).*ones(length(f{iTSCount}),1) - f{iTSCount};
    end
    disp('Cost vector done.');

    
    % Build the association/equality constraints
    A_assoc = []; A_trans = [];
    vectorSize = size(f{1},1);
    for c = 1:numCPairs
        n1 = CPairs(c,1);
        n2 = CPairs(c,2);  
        
        numP1 = numPersons(n1);
        numP2 = numPersons(n2);
        
        tmin = transitions(c);
        
        % Take track data or current pair
        id1_begin = pairwise.begin_l{n1,n2};
        id1_end = pairwise.end_l{n1,n2};
        id2_begin = pairwise.begin_c{n1,n2};
        id2_end = pairwise.end_c{n1,n2};
        
        % ADD THE TRANSITION CONSTRAINT
        for i = 1:numP1
            v = zeros(1, vectorSize);
            for j = 1:numP2

                % Transition constraint
                if id1_begin(i) < id2_begin(j)
                    diff = id2_begin(j) - id1_end(i);
                    if diff > tmin
                        assoc = 0;
                    else
                        assoc = 1;
                    end
                    
                else
                    diff = id1_begin(i) - id2_end(j);
                    if diff > tmin
                        assoc = 0;
                    else
                        assoc = 1;
                    end
                    
                end
                
                v(1, sum(vectorSize_perpair(1:c-1))+(j-1)*numP1+i) = assoc;
            end
            A_trans = [A_trans; v];
        end
        b_trans = zeros(size(A_trans,1), 1);
        
        % Compute the upper half of the matrix as given in equation (7) of the
        % supplementary. Each 'i' fills a row.
        for i = 1:numP1
            v = zeros(1, vectorSize);
            for j = 1:numP2
                
                v(1, sum(vectorSize_perpair(1:c-1))+(j-1)*numP1+i) = 1;
                
            end
            A_assoc = [A_assoc; v];
        end
        % Compute the lower half of the matrix as given in equation (7) of the
        % supplementary. Each 'j' fills a row.
        for j = 1:numP2
            v = zeros(1, vectorSize);
            for i = 1:numP1
                
                v(1, sum(vectorSize_perpair(1:c-1))+(j-1)*numP1+i) = 1;
                
            end
            A_assoc = [A_assoc; v];
        end
    end
    b_assoc = ones(size(A_assoc,1),1);
    disp('Association constraint matrix done.');

    % Get all possible triplets that can be formed out of the camera numbers.
    % Remember that this is not all possible triplets that can be formed by the
    % labels (x in paper), this is just all possible triplets that can be
    % formed by the camera numbers
    triplets = [];
    vectorSize_pertriplet = [];
    for r = 1:size(CPairs,1)
        cp = CPairs(r,1);
        cq = CPairs(r,2);
        others = setdiff(cameras,[cp,cq]);
        all_perms_tri = combnk(others, 1);
        for cr = all_perms_tri'
            triplets = [triplets; [cp,cr,cq]];
            vectorSize_pertriplet = [vectorSize_pertriplet; ...
                [numPersons(cp)*numPersons(cq)*numPersons(cr)]];
        end;
    end;
    
    
    % Fill up the loop/inequality constraints
    
% A_loop = []; b_loop = [];

    A_loop = zeros(sum(vectorSize_pertriplet), vectorSize);
    b_loop = zeros(size(A_loop,1), 1);
    for t = 1:size(triplets,1)
        % Camera p
        cp = triplets(t,1);
        % Camera r
        cr = triplets(t,2);
        % Camera q
        cq = triplets(t,3);
        
        % In 'CPairs' find the row number which contains the pair p-r
        cpr = find((CPairs(:,1)==cp & CPairs(:,2)==cr) | (CPairs(:,1)==cr & CPairs(:,2)==cp));
        
        numPp = numPersons(cp);
        numPr = numPersons(cr);
        
        % In 'CPairs' find the row number which contains the pair r-q
        crq = find((CPairs(:,1)==cr & CPairs(:,2)==cq) | (CPairs(:,1)==cq & CPairs(:,2)==cr));
        
        numPq = numPersons(cq);
        numPr = numPersons(cr);
        
        % In 'CPairs' find the row number which contains the pair p-q
        cpq = find((CPairs(:,1)==cp & CPairs(:,2)==cq) | (CPairs(:,1)==cq & CPairs(:,2)==cp));
        
        numPp = numPersons(cp);
        numPq = numPersons(cq);
        
        for i = 1:numPp
            for k = 1:numPq
                for j = 1:numPr
                    v = zeros(1, vectorSize);
                    if cp < cr 
                        v(1, sum(vectorSize_perpair(1:cpr-1))+(j-1)*numPp+i ) = 1;
                    else
                        v(1, sum(vectorSize_perpair(1:cpr-1))+(i-1)*numPr+j ) = 1;
                    end;
                    
                    if cr < cq
                        v(1, sum(vectorSize_perpair(1:crq-1))+(k-1)*numPr+j ) = 1;
                    else
                        v(1, sum(vectorSize_perpair(1:crq-1))+(j-1)*numPq+k ) = 1;
                    end;
                    
                    if cp < cq
                        v(1, sum(vectorSize_perpair(1:cpq-1))+(k-1)*numPp+i ) = -1;

                    else
                        v(1, sum(vectorSize_perpair(1:cpq-1))+(i-1)*numPq+k ) = -1; 

                    end;
                    A_loop(sum(vectorSize_pertriplet(1:t-1)) + (i-1)*numPq*numPr + (k-1)*numPr + j, :) = v;
                    b_loop(sum(vectorSize_pertriplet(1:t-1)) + (i-1)*numPq*numPr + (k-1)*numPr + j, :) = 1;
                end;
            end;
        end;
    end;
    disp('Loop constraint matrix done.');

    % Now all the constraints are inequality, so club them
    A = [A_assoc; A_loop; A_trans];
    b = [b_assoc; b_loop; b_trans];
    
    % Definition modele
    model.A = sparse(A);
    model.rhs = b;
    model.sense = '<';
    model.vtype = 'B';
    model.modelsense = 'min';
    
    % Definition params
    params.outputflag = 0;
    
%% Run integer program to solve for the labels

    ReshapingIndices = [0; cumsum(vectorSize_perpair)];
    percentageAccuracy = zeros(size(CPairs,1),length(testset));
    fprintf('=====================\n');
    for iTSCount=testset
        % !! Call gurobi ip solver !!
        model.obj = f{iTSCount};
        res = gurobi(model,params);
        x = res.x;
% % %     Cplex :   x = cplexbilp(f{iTSCount},A,b,[],[]);
        fprintf('\nkk = %0.2f Testset %d Optimization done in %.2f(s)\n', kk(kCount), iTSCount, res.runtime);
        
        for i=1:size(CPairs,1)
            c1 = CPairs(i,1);
            c2 = CPairs(i,2);
            
            numP1 = numPersons(c1);
            numP2 = numPersons(c2);
            
            Allx{c1,c2}(:,:,iTSCount) = reshape(x(ReshapingIndices(i)+1:ReshapingIndices(i+1)),...
                numP1,numP2); % !!!!
            
            % Calculate the number of correct matches
            correctMatch{c1,c2} = sum(diag(Allx{c1,c2}(1:numPers_CPair(i),1:numPers_CPair(i),iTSCount)));%(:,:,iTSCount)));
            % Calculate the number of correct 'no matches'
            CorrectNoMatch{c1,c2} = 0;
            
            if numPers_CPair(i) < numP1
                RowsCorrToExtraPeople = Allx{c1,c2}(numPers_CPair(i)+1:numP1,:,iTSCount);
                [l,~] = find(RowsCorrToExtraPeople);
                CorrectNoMatch{c1,c2} = (numP1-numPers_CPair(i))-length(unique(l));
            end
            if numPers_CPair(i) < numP2
                ColsCorrToExtraPeople = Allx{c1,c2}(:,numPers_CPair(i)+1:numP2,iTSCount);
                [~,c] = find(ColsCorrToExtraPeople);
                CorrectNoMatch{c1,c2} = CorrectNoMatch{c1,c2} + (numP2-numPers_CPair(i))-length(unique(c));
            end
            
            correctResult = correctMatch{c1,c2} + CorrectNoMatch{c1,c2};
            percentageAccuracy(i,iTSCount) = correctResult/(numPers_CPair(i)+numP1-numPers_CPair(i)+numP2-numPers_CPair(i))*100;
            fprintf('Camera %d-%d gave true positive %d ; true negative %d. Accuracy = %2.2f%%\n',...
                   c1,c2,correctMatch{c1,c2},CorrectNoMatch{c1,c2},percentageAccuracy(i,iTSCount))
        end
     end
    fprintf('\n');
    
    % Average Accuracy
    AvgPercentageAccuracy = mean(percentageAccuracy,2);
    for i=1:size(CPairs,1)
        c1 = CPairs(i,1);
        c2 = CPairs(i,2);
       fprintf('Camera %d-%d gave average accuracy = %2.2f%%\n',...
                   c1,c2,AvgPercentageAccuracy(i,1))
    end
    
    res_AvgPercentageAccuracy(:,kCount) = AvgPercentageAccuracy;

    fprintf('\n\n');
    % Save x's
    save(['NCR_Code-master/Results/Res_' dataset_name '_avec_contrainte_t.mat'],'Allx','correctMatch','percentageAccuracy');

end
%%
nCurve = size(res_AvgPercentageAccuracy,1);
figure;
couleurs = hsv(nCurve);
for p=1:nCurve ;
    h(p) = line(kk,res_AvgPercentageAccuracy(p,:), 'color', couleurs(p,:)); title('Re-ID accurancy with varying k');

end
legend(h, num2str(CPairs));



%%
%load('NCR_Code-master/Results/Res_camnet.mat');
















%% liens utiles
% https://www.hpc.science.unsw.edu.au/files/docs/ilog/cplex/12.1/html/Content/Optimization/Documentation/CPLEX/_pubskel/cplex_matlab1210.html
% https://www.gurobi.com/documentation/6.5/refman/matlab_gurobi.html